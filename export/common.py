"""Shared settings and loaders of the export pipeline (host side, needs torch and a GPT-SoVITS checkout)."""
import contextlib
import io
import os
import sys
from pathlib import Path

# Static shapes of the NPU graphs. The device runtime reads them back from manifest.json.
P = 512  # t2s prefill length: reference phones + target phones + reference semantic tokens
L = 1024  # t2s kv-cache length (P + FR // 2 = 762 would also fit in 768: measured only 15% faster per step)
FR = 500  # vits frames (50 Hz) per segment -> 10 s, i.e. at most FR // 2 = 250 semantic tokens
TT = 128  # vits phones per segment
WIN = 96  # vits generator window (frames); HiFiGAN over 500 frames needs 27 MB of TCM (8 MB on v73)
OV = 12  # generator window overlap on each side (frames)
BERT_S = 128  # BERT tokens per segment
BERT_LAYERS = 22  # GPT-SoVITS uses hidden_states[-3] of the 24-layer RoBERTa = output of layer 22
G2PW_B, G2PW_S = 8, 64  # g2pw rows (polyphonic chars) x tokens per call

GRAPHS = {  # graph key -> file stem
    "g2pw": f"g2pw_B{G2PW_B}_S{G2PW_S}",
    "bert": f"bert{BERT_LAYERS}_S{BERT_S}",
    "t2s_prefill": f"t2s_prefill_P{P}",
    "t2s_decode": f"t2s_decode_L{L}",
    "vits_enc": f"vits_enc_F{FR}_T{TT}",
    "vits_gen": f"vits_gen_W{WIN}",
}


def setup_gsv(gsv_root):
    """Make the GPT-SoVITS sources importable (module/, AR/, text/, ...)."""
    root = Path(gsv_root).resolve()
    if not (root / "GPT_SoVITS" / "module" / "models_onnx.py").exists():
        raise SystemExit(f"{root} is not a GPT-SoVITS checkout (GPT_SoVITS/module/models_onnx.py missing)")
    for p in (str(root / "GPT_SoVITS"), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


@contextlib.contextmanager
def cwd(path):
    """Some GPT-SoVITS modules resolve model paths relative to the working directory (e.g. sv.py)."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@contextlib.contextmanager
def _windows_paths():
    """Lightning checkpoints written on Windows pickle WindowsPath objects (hyper_parameters.output_dir)."""
    import pathlib

    saved = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath
    try:
        yield
    finally:
        pathlib.WindowsPath = saved


def load_t2s(ckpt):
    """GPT (text-to-semantic) weights -> (config, fp32 state dict with 'model.' prefixed keys). Accepts the
    training checkpoint logs/<exp>/logs_s1_*/ckpt/epoch=N-step=M.ckpt (= GPT_weights e<N+1> in fp32) as well as
    the exported GPT_weights*/<exp>-eN.ckpt."""
    import torch

    with _windows_paths():
        d = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    if "state_dict" in d:  # Lightning training checkpoint
        return d["hyper_parameters"]["config"], {k: v.float() for k, v in d["state_dict"].items()}
    return d["config"], {k: v.float() for k, v in d["weight"].items()}


def load_sovits_raw(pth, config_json=None):
    """SoVITS weights -> {"weight": state dict, "config": hparams}. Accepts the training checkpoint
    logs/<exp>/logs_s2_*/G_<step>.pth (config from logs/<exp>/config.json) and the exported SoVITS_weights*/*.pth
    (whose first two bytes are a version tag instead of 'PK')."""
    import json

    import torch
    from utils import HParams  # noqa: F401  (needed to unpickle the config of exported weights)

    b = open(pth, "rb").read()
    d = torch.load(io.BytesIO(b if b[:2] == b"PK" else b"PK" + b[2:]), map_location="cpu", weights_only=False)
    if "model" in d and "config" not in d:  # G_<step>.pth training checkpoint
        cj = config_json or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(pth))), "config.json")
        return {"weight": d["model"], "config": json.load(open(cj)), "iteration": d.get("iteration")}
    return d


def load_vits(pth, config_json=None):
    """-> (SynthesizerTrn from module/models_onnx.py in eval mode, weight norm removed, data hparams dict)."""
    from module.models_onnx import SynthesizerTrn

    d = load_sovits_raw(pth, config_json)
    to = lambda x: x if isinstance(x, dict) else x.__dict__  # noqa: E731
    h = to(d["config"])
    hm, hd, ht = to(h["model"]), to(h["data"]), to(h["train"])
    if hm.get("version") not in ("v2Pro", "v2ProPlus"):
        raise SystemExit(f"{pth}: SoVITS version {hm.get('version')!r}; this pipeline supports v2Pro / v2ProPlus")
    m = SynthesizerTrn(hd["filter_length"] // 2 + 1, ht["segment_size"] // hd["hop_length"],
                       n_speakers=hd["n_speakers"], **hm)
    m.load_state_dict({k: v.float() for k, v in d["weight"].items()}, strict=False)
    m.eval()
    m.dec.remove_weight_norm()
    return m, hd


# ------------------------------------------------------------------ training experiment (logs/<exp>)
def experiment(exp_dir, gpt_epoch=None, sovits_epoch=None):
    """Resolve a GPT-SoVITS training folder (copied from GPT-SoVITS/logs/<exp>) to the checkpoints to convert.
    Epochs are numbered like the exported weights: GPT epoch=N-step=M.ckpt is eN+1, SoVITS G_<step>.pth is
    e<iteration>. Default: the last epoch of each (export/eval_sovits_ckpts.py helps to pick a SoVITS epoch)."""
    import glob
    import re

    import torch

    exp = Path(exp_dir).resolve()
    gpts = {int(re.search(r"epoch=(\d+)", f).group(1)) + 1: f
            for f in glob.glob(str(exp / "logs_s1*" / "ckpt" / "epoch=*-step=*.ckpt"))}
    sov = {}
    for f in glob.glob(str(exp / "logs_s2*" / "G_*.pth")):
        sov[int(torch.load(f, map_location="cpu", weights_only=False, mmap=True).get("iteration"))] = f
    if not gpts or not sov:
        raise SystemExit(f"{exp}: no GPT (logs_s1*/ckpt/epoch=*.ckpt) or SoVITS (logs_s2*/G_*.pth) checkpoints")
    ge, se = gpt_epoch or max(gpts), sovits_epoch or max(sov)
    if ge not in gpts or se not in sov:
        raise SystemExit(f"{exp.name}: available epochs: GPT {sorted(gpts)}, SoVITS {sorted(sov)}")
    return {"name": exp.name, "dir": str(exp), "gpt": gpts[ge], "sovits": sov[se], "gpt_epoch": ge,
            "sovits_epoch": se, "gpt_epochs": sorted(gpts), "sovits_epochs": sorted(sov), "sovits_files": sov}


def add_model_args(ap):
    ap.add_argument("--gsv-root", default=os.environ.get("GSV_ROOT"),
                    help="GPT-SoVITS directory (code + GPT_SoVITS/pretrained_models); default: $GSV_ROOT")
    ap.add_argument("--exp", help="training folder copied from GPT-SoVITS/logs/<exp> (e.g. logs/myvoice)")
    ap.add_argument("--gpt-epoch", type=int, help="GPT epoch to convert (default: the last one)")
    ap.add_argument("--sovits-epoch", type=int, help="SoVITS epoch to convert (default: the last one)")
    ap.add_argument("--gpt-ckpt", help="instead of --exp: exported GPT_weights*/*.ckpt")
    ap.add_argument("--sovits-pth", help="instead of --exp: exported SoVITS_weights*/*.pth")


def resolve_models(a):
    """Fill a.gpt_ckpt / a.sovits_pth (absolute) from --exp or the explicit paths; returns the GPT-SoVITS root."""
    if not a.gsv_root:
        raise SystemExit("set --gsv-root or GSV_ROOT to your GPT-SoVITS directory")
    root = setup_gsv(a.gsv_root)
    a.exp_info = None
    if a.exp:
        a.exp_info = experiment(a.exp, a.gpt_epoch, a.sovits_epoch)
        a.gpt_ckpt = a.gpt_ckpt or a.exp_info["gpt"]
        a.sovits_pth = a.sovits_pth or a.exp_info["sovits"]
        print(f"[{a.exp_info['name']}] GPT e{a.exp_info['gpt_epoch']} (available {a.exp_info['gpt_epochs']}), "
              f"SoVITS e{a.exp_info['sovits_epoch']} (available {a.exp_info['sovits_epochs']})", flush=True)
    if not (a.gpt_ckpt and a.sovits_pth):
        raise SystemExit("give --exp logs/<exp>, or --gpt-ckpt and --sovits-pth")
    a.gpt_ckpt, a.sovits_pth = os.path.abspath(a.gpt_ckpt), os.path.abspath(a.sovits_pth)
    return root


def pick_reference(exp_dir, name=None, target_s=6.0):
    """A training clip of the experiment as reference voice -> (wav path, transcript, seconds). Default: the clip
    of 3..10 s whose duration is closest to target_s (the reference also takes prefill positions: ~25 tokens/s)."""
    import wave

    exp = Path(exp_dir)
    texts = {}
    for line in (exp / "2-name2text.txt").read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) >= 4:
            texts[p[0]] = p[3]
    best = None
    for n in sorted(texts):
        w = exp / "5-wav32k" / n
        if not w.exists() or (name and n not in (name, name + ".wav")):
            continue
        with wave.open(str(w)) as f:
            dur = f.getnframes() / f.getframerate()
        if (name or 3.0 <= dur <= 10.0) and (best is None or abs(dur - target_s) < abs(best[2] - target_s)):
            best = (str(w), texts[n], dur)
    if best is None:
        raise SystemExit(f"no usable reference clip in {exp / '5-wav32k'}" + (f" named {name}" if name else ""))
    return best


def save_onnx_dir(model, out_dir):
    """Save as AI Hub's directory format (name.onnx/{model.onnx, model.data}); drop graph IO names from
    value_info (onnxslim writes them there and AI Hub rejects that: "occur in value_info but also in model IO")."""
    import onnx

    io_names = {x.name for x in list(model.graph.input) + list(model.graph.output)}
    for v in [v for v in model.graph.value_info if v.name in io_names]:
        model.graph.value_info.remove(v)
    os.makedirs(out_dir, exist_ok=True)
    onnx.save(model, os.path.join(out_dir, "model.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
              location="model.data")


def export_torch(module, args, input_names, output_names, out_dir, slim=True):
    import onnx
    import torch

    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "tmp.onnx")
    with torch.no_grad():
        torch.onnx.export(module.eval(), args, tmp, input_names=input_names, output_names=output_names,
                          opset_version=17, do_constant_folding=True)
    m = onnx.load(tmp)
    if slim:
        import onnxslim

        m = onnxslim.slim(m)
    save_onnx_dir(m, out_dir)
    for f in os.listdir(out_dir):
        if f.startswith("tmp.onnx") or (f not in ("model.onnx", "model.data")):
            os.remove(os.path.join(out_dir, f))
