"""GPT-SoVITS v2Pro runtime for Qualcomm QCS8550 (HTP / NPU), numpy only on the CPU side.

    text --(g2p: jieba + [g2pw] polyphones)--> phones --[bert]--> phone features
         --[t2s_prefill] + N x [t2s_decode] (sampling / KV cache on CPU)--> semantic tokens
         --[vits_enc] + [vits_gen] x windows--> 32 kHz audio

[...] = static-shape graph on the NPU (see manifest.json). Two interchangeable backends:
    ort : onnxruntime on the fp32 ONNX graphs (PC testing, same graphs as the device)
    qnn : QNN context binaries (models/*.bin) through lib/libgsv_qnn.so (QNN C API, via ctypes)

Reference voices are precomputed on a PC (export/prepare_assets.py) into assets/ref_<name>.npz.
Only Chinese text is supported (frontend = GPT-SoVITS chinese2 + G2PW). Static shapes and model dimensions come
from manifest.json ("config", written by export/make_deploy.py).

Usage:
    python gsv_runtime.py --backend ort --text "今天天气不错。" --out out.wav
    source setup_env.sh && python gsv_runtime.py --backend qnn --text ... --out out.wav
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import wave
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # -> ./text (frontend package)
os.environ.setdefault("bert_path", os.path.join(HERE, "assets", "bert_tokenizer"))

# defaults; configure(manifest) overrides them with the values the graphs were exported with
SR = 32000
HOP = 640  # audio samples per 50 Hz frame
EOS = 1024
KV = (24, 16, 32)  # t2s layers, heads, head dim
NEG = -100.0  # finite, small-range mask value: same result as -1e4, keeps int16 activation ranges tight
P, L = 512, 1024  # t2s prefill length / kv-cache length
FR, TT = 500, 128  # vits frames (10 s) / vits phones
BERT_S = 128
WIN, OV = 96, 12  # vits generator window / overlap (frames)
MIN_TOKENS_BEFORE_EOS = 11
PUNCT = "，。？！,.?!~…、；;：:”」』’）)\"'"


# ------------------------------------------------------------------ backends
def graphs_of(manifest):
    """Graph entries of manifest.json (everything except the "config" section)."""
    return {g: v for g, v in manifest.items() if g != "config"}


class OrtBackend:
    def __init__(self, manifest, onnx_dir):
        self.full_manifest, manifest = manifest, graphs_of(manifest)
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.man = manifest
        self.sess = {g: ort.InferenceSession(os.path.join(onnx_dir, v["file"][:-4] + ".onnx", "model.onnx"), so,
                                             providers=["CPUExecutionProvider"]) for g, v in manifest.items()}

    def run(self, g, feeds):
        names = [o[0] for o in self.man[g]["outputs"]]
        return dict(zip(names, self.sess[g].run(None, feeds)))


QNN_DTYPES = {0x0232: np.float32, 0x0216: np.float16, 0x0032: np.int32, 0x0132: np.uint32, 0x0064: np.int64,
              0x0008: np.int8, 0x0108: np.uint8}


class QnnBackend:
    """QNN HTP through lib/libgsv_qnn.so (QNN C API; loading path validated on a QCS8550 board).

    Tensor names/order/dtypes are read from each context binary: AI Hub keeps input names but may reorder them,
    and renames outputs to output_0..N in ONNX output order. Inputs are passed zero-copy (the numpy buffers are
    handed to QNN); outputs land in per-graph buffers that are reused by the next call of the same graph."""

    def __init__(self, manifest, models_dir, qnn_lib_dir, lib_path=None, burst=True, log_level=1):
        import ctypes

        self.ct = ctypes
        self.lib = L = ctypes.CDLL(lib_path or os.path.join(HERE, "lib", "libgsv_qnn.so"))
        vp, cp = ctypes.c_void_p, ctypes.c_char_p
        L.gq_create.restype = vp
        L.gq_create.argtypes = [cp, cp, ctypes.POINTER(cp), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.gq_last_error.restype = cp
        L.gq_num_graphs.argtypes = L.gq_grouped.argtypes = L.gq_destroy.argtypes = [vp]
        L.gq_num_tensors.argtypes = [vp, ctypes.c_int, ctypes.c_int]
        L.gq_tensor_info.argtypes = [vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint32),
                                     ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulonglong)]
        L.gq_execute.argtypes = [vp, ctypes.c_int, ctypes.POINTER(vp), ctypes.POINTER(vp), ctypes.POINTER(ctypes.c_double)]
        self.full_manifest = manifest
        self.man = manifest = graphs_of(manifest)
        self.names = list(manifest)  # graph index = manifest order (one single-graph binary per entry)
        paths = [os.path.join(models_dir, manifest[g]["file"]) for g in self.names]
        arr = (cp * len(paths))(*[p.encode() for p in paths])
        self.h = L.gq_create(os.path.join(qnn_lib_dir, "libQnnHtp.so").encode(),
                             os.path.join(qnn_lib_dir, "libQnnSystem.so").encode(), arr, len(paths),
                             1 if burst else 0, log_level)
        if not self.h:
            raise RuntimeError(f"gq_create failed: {L.gq_last_error().decode()}")
        if L.gq_num_graphs(self.h) != len(self.names):
            raise RuntimeError(f"expected {len(self.names)} graphs, binaries contain {L.gq_num_graphs(self.h)}")
        self.io = {g: self._describe(i, g) for i, g in enumerate(self.names)}
        self.last_ms = 0.0
        # Free contexts/device/backend while the process is still intact: left to process teardown, the HTP
        # objects outlive the FastRPC session and exit crashes ("undefined m_mutex handle object", segfault).
        import atexit

        atexit.register(self.close)

    def _tensors(self, gi, is_out):
        ct, out = self.ct, []
        for i in range(self.lib.gq_num_tensors(self.h, gi, int(is_out))):
            name, dt, rank, nbytes = ct.create_string_buffer(256), ct.c_int(), ct.c_int(), ct.c_ulonglong()
            dims = (ct.c_uint32 * 8)()
            if self.lib.gq_tensor_info(self.h, gi, int(is_out), i, name, 256, ct.byref(dt), dims, ct.byref(rank),
                                       ct.byref(nbytes)) != 0:
                raise RuntimeError(self.lib.gq_last_error().decode())
            if dt.value not in QNN_DTYPES:
                raise RuntimeError(f"unsupported QNN dtype 0x{dt.value:04x} for tensor {name.value.decode()}")
            out.append((name.value.decode(), tuple(dims[: rank.value]), QNN_DTYPES[dt.value], nbytes.value))
        return out

    def _describe(self, gi, g):
        spec = self.man[g]
        ins, outs = self._tensors(gi, False), self._tensors(gi, True)
        want = {n: tuple(s) for n, s, _ in spec["inputs"]}
        got = {n: s for n, s, _, _ in ins}
        if want != got:
            raise RuntimeError(f"{g}: binary inputs {got} do not match manifest {want}")
        # outputs: output_<k> is the k-th ONNX output
        onames = [n for n, _ in spec["outputs"]]
        order = sorted(range(len(outs)), key=lambda i: int(outs[i][0].rsplit("_", 1)[-1]) if outs[i][0].startswith("output_") else i)
        bufs = [np.empty(s, t) for _, s, t, _ in outs]
        mapping = {onames[rank]: bufs[i] for rank, i in enumerate(order)}
        return ins, outs, bufs, mapping

    def run(self, g, feeds):
        ct = self.ct
        ins, outs, bufs, mapping = self.io[g]
        keep = [np.ascontiguousarray(feeds[n], dtype=t) for n, s, t, _ in ins]  # zero-copy when already matching
        for (n, s, _, nb), a in zip(ins, keep):
            if a.shape != s or a.nbytes != nb:
                raise ValueError(f"{g}.{n}: got {a.shape}, graph expects {s}")
        in_p = (ct.c_void_p * len(keep))(*[a.ctypes.data for a in keep])
        out_p = (ct.c_void_p * len(bufs))(*[b.ctypes.data for b in bufs])
        ms = ct.c_double()
        if self.lib.gq_execute(self.h, self.names.index(g), in_p, out_p, ct.byref(ms)) != 0:
            raise RuntimeError(f"gq_execute({g}) failed: {self.lib.gq_last_error().decode()}")
        self.last_ms = ms.value
        return {n: (b if b.dtype == np.float32 else b.astype(np.float32)) for n, b in mapping.items()}

    def close(self):
        if getattr(self, "h", None):
            self.lib.gq_destroy(self.h)
            self.h = None


G2PW_B, G2PW_S = 8, 64  # static g2pw graph: rows (polyphonic chars) x tokens


def configure(manifest):
    """Take static shapes / model dimensions from manifest["config"] (see export/make_deploy.py)."""
    global SR, HOP, EOS, KV, P, L, FR, TT, BERT_S, WIN, OV, G2PW_B, G2PW_S
    c = manifest.get("config")
    if not c:
        return
    sh, t = c["shapes"], c["t2s"]
    SR, HOP, EOS = c["sampling_rate"], c["hop_length"], t["vocab"] - 1
    KV = (t["layers"], t["heads"], t["dim"] // t["heads"])
    P, L, FR, TT, BERT_S = sh["P"], sh["L"], sh["FR"], sh["TT"], sh["BERT_S"]
    WIN, OV, G2PW_B, G2PW_S = sh["WIN"], sh["OV"], sh["G2PW_B"], sh["G2PW_S"]


class G2pwHead:
    """G2PW output head in numpy fp32 (the NPU graph stops at the hidden state of the query char).
    fp16 is unsafe here: the masked softmax exp(logit - global max) underflows for the allowed readings and the
    POS ArgMax flips on near-ties, which changed ~4% of the predictions when the head ran on the HTP."""

    def __init__(self, d):
        ld = lambda n, mm=None: np.load(os.path.join(d, n + ".npy"), mmap_mode=mm)  # noqa: E731
        self.pos_w, self.pos_b, self.cls_w, self.cls_b = ld("pos_w"), ld("pos_b"), ld("cls_w"), ld("cls_b")
        self.desc_bias = ld("desc_bias")
        self.char_desc, self.second_order = ld("char_desc"), ld("second_order")  # 19 + 205 MB, loaded into RAM
        self.n_pos = self.pos_w.shape[0]

    def __call__(self, hidden, feeds):
        h = hidden.astype(np.float32)
        char = np.clip(feeds["char_ids"].astype(np.int64), 0, self.char_desc.shape[0] - 1)
        pos = np.argmax(h @ self.pos_w.T + self.pos_b, axis=-1)
        logits = h @ self.cls_w.T + self.cls_b
        desc = self.desc_bias + self.char_desc[char] + self.second_order[char * self.n_pos + pos]
        mask = feeds["phoneme_mask"] / (1.0 + np.exp(-desc))
        e = np.exp(logits - logits.max(-1, keepdims=True)) * mask
        return np.clip(e / e.sum(-1, keepdims=True), 1e-6, 0.999999).astype(np.float32)


class NpuG2pw:
    """Drop-in for the G2PW onnxruntime session (text/g2pw/onnx_api.py SESSION_FACTORY): pads the dynamic
    (rows, tokens) batch to the static g2pw graph, runs it in chunks of G2PW_B rows on the backend."""

    def __init__(self, backend, head_dir=os.path.join(HERE, "assets", "g2pw_head"), head=None):
        self.b = backend
        self.head = head or G2pwHead(head_dir)

    def run(self, output_names, feeds):
        n, L = feeds["input_ids"].shape
        ids, tt, am = feeds["input_ids"], feeds["token_type_ids"], feeds["attention_mask"]
        pos = feeds["position_ids"].astype(np.int64)
        if L > G2PW_S:  # rare (clauses are short): keep a window of G2PW_S tokens around each query char
            st = np.clip(pos - G2PW_S // 2, 0, L - G2PW_S)
            take = st[:, None] + np.arange(G2PW_S)
            ids, tt, am = (np.take_along_axis(x, take, 1) for x in (ids, tt, am))
            pos, L = pos - st, G2PW_S
        probs = []
        for s in range(0, n, G2PW_B):
            k = min(G2PW_B, n - s)
            rows = np.r_[np.arange(s, s + k), np.full(G2PW_B - k, s)]  # pad rows repeat a real row, ignored
            f = {}
            for name, x in (("input_ids", ids), ("token_type_ids", tt), ("attention_mask", am)):
                a = np.zeros((G2PW_B, G2PW_S), np.int32)
                a[:, :L] = x[rows]
                f[name] = a
            f["phoneme_mask"] = np.ascontiguousarray(feeds["phoneme_mask"][rows], np.float32)
            f["char_ids"] = feeds["char_ids"][rows].astype(np.int32)
            f["position_ids"] = pos[rows].astype(np.int32)
            npu = {n: f[n] for n in ("input_ids", "token_type_ids", "attention_mask", "position_ids")}
            probs.append(self.head(self.b.run("g2pw", npu)["hidden"], f)[:k])
        return [np.concatenate(probs)]


class Timed:
    """Wraps a backend, accumulates per-graph call counts and wall time."""

    def __init__(self, backend):
        self.b = backend
        self.calls = defaultdict(int)
        self.secs = defaultdict(float)

    def run(self, g, feeds):
        t = time.perf_counter()
        out = self.b.run(g, feeds)
        self.calls[g] += 1
        self.secs[g] += time.perf_counter() - t
        return out


# ------------------------------------------------------------------ text
SENT_END = "。！？!?…"  # sentence-ending punctuation (a "." between digits is a decimal point, not an end)
SUB_SPLIT = "，,、；;：:"  # used only to cut sentences longer than maxc
_TN = None


def han_len(s):
    """Length in Chinese characters after the frontend's text normalization (numbers/dates/units expanded:
    "2026年" -> 二零二六年), i.e. what the model actually reads; punctuation is not counted."""
    global _TN
    if _TN is None:
        from text.zh_normalization.text_normlization import TextNormalizer

        _TN = TextNormalizer()
    return len(re.findall(r"[\u4e00-\u9fa5]", "".join(_TN.normalize(s)))) if s.strip() else 0


def _cut_long(piece, maxc):
    """Cut a sentence longer than maxc: pack comma/semicolon clauses greedily, hard-cut clauses still too long."""
    out, cur = [], ""
    for c in re.split(f"(?<=[{SUB_SPLIT}])", piece):
        while han_len(c) > maxc:  # no usable punctuation: longest prefix that fits
            k = max(i for i in range(1, len(c) + 1) if han_len(c[:i]) <= maxc)
            if cur:
                out.append(cur)
                cur = ""
            out.append(c[:k])
            c = c[k:]
        if cur and han_len(cur + c) > maxc:
            out.append(cur)
            cur = c
        else:
            cur += c
    if cur:
        out.append(cur)
    return out


def split_text(text, maxc=30, minc=5):
    """Split at sentence-ending punctuation (。！？!?… and line breaks). Sentences longer than maxc characters are
    cut at commas/semicolons (hard cut if there are none); segments shorter than minc characters are merged into
    a neighbour when the result stays <= maxc. Lengths are counted by han_len (after number normalization)."""
    text = re.sub(f"(?<=[{SENT_END}”」』’）)\"'])\\s*\\n+\\s*", "", text.strip())  # line break after an end mark
    text = re.sub(r"\s*\n+\s*", "。", text)  # any other line break ends a sentence
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", "。", text)  # "." ends a sentence unless it is a decimal point
    parts = re.split(f"([{SENT_END}]+[”」』’）)\"']*)", text)
    sents = ["".join(parts[i:i + 2]).strip() for i in range(0, len(parts), 2)]
    pieces = []
    for s in sents:
        if not s:
            continue
        if han_len(s) == 0:  # punctuation only: attach to the previous piece
            if pieces:
                pieces[-1] += s
            continue
        pieces += _cut_long(s, maxc) if han_len(s) > maxc else [s]
    i = 0  # merge short pieces into a neighbour (previous first) while staying <= maxc
    while i < len(pieces):
        n = han_len(pieces[i])
        if n < minc and len(pieces) > 1:
            if i > 0 and han_len(pieces[i - 1] + pieces[i]) <= maxc:
                pieces[i - 1] += pieces.pop(i)
                continue
            if i + 1 < len(pieces) and han_len(pieces[i] + pieces[i + 1]) <= maxc:
                pieces[i] += pieces.pop(i + 1)
                continue
        i += 1
    return [p if p[-1] in PUNCT else p + "。" for p in pieces]


NOTCH_HZ = (11000.0, 13000.0, 13500.0, 14500.0, 15700.0)  # tonal comb left by SoVITS fine-tuning (500 Hz multiples)


def post_filter(audio, sr=SR, mode="off", presence_db=0.0, notch_q=30.0, lp_hz=12000.0, lp_order=8):
    """Zero-phase spectral post-processing (numpy FFT; magnitude = that of the matching scipy filtfilt design).
    mode: "off"     - nothing
          "notch"   - narrow notches at NOTCH_HZ only (removes the fine-tuning whine, keeps the 12-16 kHz air)
          "lowpass" - 8th-order 12 kHz low-pass + 11 kHz notch (removes the whine and the air: sounds duller)
    presence_db: high-shelf gain from ~4 kHz up (the model is 5-6 dB short of the recordings at 4-10 kHz)."""
    n = len(audio)
    if n == 0 or (mode == "off" and not presence_db):
        return audio
    pad = 4096  # keep the circular FFT convolution away from the signal edges
    spec = np.fft.rfft(np.concatenate([audio, np.zeros(pad, audio.dtype)]))
    f = np.fft.rfftfreq(n + pad, 1.0 / sr)
    w = np.tan(np.pi * f / sr)  # bilinear-transform frequency warping
    h = np.ones_like(f)

    def notch2(hz):
        w0 = np.tan(np.pi * hz / sr)
        num = (w ** 2 - w0 ** 2) ** 2
        return num / (num + (w * w0 / notch_q) ** 2 + 1e-30)

    if mode == "lowpass":
        h *= 1.0 / (1.0 + (w / np.tan(np.pi * lp_hz / sr)) ** (2 * lp_order)) * notch2(11000.0)
    elif mode == "notch":
        for hz in NOTCH_HZ:
            h *= notch2(hz)
    elif mode != "off":
        raise ValueError(f"unknown filter mode {mode!r}")
    if presence_db:
        t = np.clip((f - 3000.0) / 3000.0, 0, 1)  # raised-cosine transition 3 -> 6 kHz
        h *= 10 ** (presence_db / 20 * (0.5 - 0.5 * np.cos(np.pi * t)))
    return np.fft.irfft(spec * h, n + pad)[:n].astype(np.float32)


def fade_edges(audio, sr=SR, in_ms=10.0, out_ms=40.0):
    """Raised-cosine fade-in/out: segments end right at the last phone (5-20 ms of tail), an abrupt drop into the
    inserted silence is heard as a cut between sentences."""
    a = audio.copy()
    for ms, sl, rev in ((in_ms, slice(None), False), (out_ms, slice(None, None, -1), True)):
        k = min(int(sr * ms / 1000), len(a) // 2)
        if k > 1:
            ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))).astype(np.float32)
            if rev:
                a[-k:] *= ramp[::-1]
            else:
                a[:k] *= ramp
    return a


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ------------------------------------------------------------------ pipeline
class GPTSoVITS:
    def __init__(self, backend, ref_npz, assets=os.path.join(HERE, "assets"), seed=None):
        """ref_npz: one reference-voice file, or a directory: every ref_<voice>.npz in it is loaded."""
        from transformers import AutoTokenizer

        self.b = backend
        configure(getattr(getattr(backend, "b", backend), "full_manifest", {}))
        self.w = dict(np.load(os.path.join(assets, "cpu_weights.npz")))
        files = sorted(glob.glob(os.path.join(ref_npz, "ref_*.npz"))) if os.path.isdir(ref_npz) else [ref_npz]
        self.refs = {}
        for f in files:
            r = np.load(f)
            self.refs[os.path.basename(f)[4:-4] if os.path.basename(f).startswith("ref_") else f] = {k: r[k] for k in r.files}
        if not self.refs:
            raise FileNotFoundError(f"no reference voice in {ref_npz}")
        self.voice = next(iter(self.refs))
        self.ref = self.refs[self.voice]
        self.tok = AutoTokenizer.from_pretrained(os.path.join(assets, "bert_tokenizer"))
        if "g2pw" in getattr(getattr(backend, "b", backend), "man", {}):  # polyphone model on the backend (NPU)
            import text.g2pw.onnx_api as g2pw_api

            g2pw_api.SESSION_FACTORY = lambda _path: NpuG2pw(backend)
        self.rng = np.random.default_rng(seed)

    # -- text -> phones, phone-level BERT features
    def frontend(self, seg):
        from text import cleaned_text_to_sequence
        from text.cleaner import clean_text

        phones, word2ph, norm = clean_text(seg, "zh", "v2")
        return np.array(cleaned_text_to_sequence(phones, "v2"), np.int64), word2ph, norm

    def bert(self, norm, word2ph):
        ids = self.tok(norm)["input_ids"]
        n = len(ids)
        assert n <= BERT_S, f"segment too long for BERT ({n} tokens > {BERT_S})"
        inp = np.zeros((1, BERT_S), np.int32)
        inp[0, :n] = ids
        mask = np.full((1, 1, 1, BERT_S), NEG, np.float32)
        mask[..., :n] = 0
        h = self.b.run("bert", {"input_ids": inp, "mask_add": mask})["h"][0, 1 : n - 1]
        assert len(h) == len(word2ph) == len(norm)
        return np.repeat(h, word2ph, axis=0)  # [n_phones, 1024]

    # -- AR semantic token generation
    def sample(self, logits, prev, top_k, top_p, temperature, rep_penalty):
        lg = logits.astype(np.float64).copy()
        idx = np.unique(prev)
        idx = idx[idx < len(lg)]
        s = lg[idx]
        lg[idx] = np.where(s < 0, s * rep_penalty, s / rep_penalty)
        argmax = int(lg.argmax())
        if top_p < 1.0:
            order = np.argsort(-lg)
            cum = np.cumsum(softmax(lg[order]))
            rm = cum > top_p
            rm[0] = False
            lg[order[rm]] = -np.inf
        lg = lg / max(temperature, 1e-5)
        if top_k:
            kth = np.sort(lg)[-min(top_k, len(lg))]
            lg[lg < kth] = -np.inf
        p = softmax(lg)
        q = self.rng.exponential(size=p.shape)
        return int(np.argmax(p / q)), argmax

    def t2s(self, phones, bert, top_k=15, top_p=1.0, temperature=1.0, rep_penalty=1.35, max_new=FR // 2):
        w, prompt = self.w, self.ref["prompt"]
        lx, ly = len(phones), len(prompt)
        n = lx + ly
        assert n <= P, f"prefill too long ({n} > {P}); use a shorter segment or reference"
        xe = w["text_emb"][phones] + bert @ w["bert_proj_w"].T + w["bert_proj_b"] + w["text_alpha"] * w["pe"][:lx]
        ye = w["audio_emb"][prompt] + w["audio_alpha"] * w["pe"][:ly]
        xy = np.zeros((1, P, 512), np.float32)
        xy[0, :n] = np.concatenate([xe, ye])
        mask = np.full((1, 1, P, P), NEG, np.float32)
        mask[0, 0, :n, :lx] = 0  # everyone sees all phones
        mask[0, 0, lx:n, lx:n] = np.where(np.tril(np.ones((ly, ly), bool)), 0, NEG)  # causal over audio
        o = self.b.run("t2s_prefill", {"xy": xy, "mask": mask})
        logits = o["h"][0, n - 1] @ w["pred_w"].T
        kT = np.zeros((KV[0], KV[1], KV[2], L), np.float32)
        vc = np.zeros((KV[0], KV[1], L, KV[2]), np.float32)
        kT[..., :n] = o["k"][:, :, :n].transpose(0, 1, 3, 2)
        vc[:, :, :n] = o["v"][:, :, :n]
        dmask = np.full((1, 1, 1, L), NEG, np.float32)
        y, pos, eos = list(prompt), n, False
        for i in range(max_new):
            lg = logits[:-1] if i < MIN_TOKENS_BEFORE_EOS else logits
            tok, am = self.sample(lg, np.array(y), top_k, top_p, temperature, rep_penalty)
            y.append(tok)
            if tok == EOS or am == EOS:
                eos = True
                break
            if pos >= L:
                break
            x = (w["audio_emb"][tok] + w["audio_alpha"] * w["pe"][ly + i])[None, None].astype(np.float32)
            dmask[..., :pos] = 0
            o = self.b.run("t2s_decode", {"x": x, "kT_cache": kT, "v_cache": vc, "mask": dmask})
            logits = o["logits"][0]
            kT[..., pos] = o["k_new"][..., 0]
            vc[:, :, pos] = o["v_new"][:, :, 0]
            pos += 1
        gen = y[ly:]
        if eos:
            gen = gen[:-1]  # same as the original: the token sampled at the stop step is dropped
        return np.array(gen, np.int64), eos

    # -- semantic tokens + phones -> audio
    def vits(self, tokens, phones, noise_scale=0.5):
        w = self.w
        f, nt = 2 * len(tokens), len(phones)
        assert f <= FR and nt <= TT, f"segment too long for vits ({f}/{FR} frames, {nt}/{TT} phones)"
        ssl = np.zeros((1, 768, FR), np.float32)
        ssl[0, :, :f] = np.repeat(w["codebook"][tokens].T, 2, axis=1)  # 25 Hz -> 50 Hz
        ym = np.zeros((1, 1, FR), np.float32)
        ym[..., :f] = 1
        te = np.zeros((1, 192, TT), np.float32)
        te[0, :, :nt] = w["vits_text_emb"][phones].T
        tm = np.zeros((1, 1, TT), np.float32)
        tm[..., :nt] = 1
        noise = (self.rng.standard_normal((1, 192, FR)) * noise_scale).astype(np.float32)
        z = self.b.run("vits_enc", {"ssl": ssl, "y_mask": ym, "text_emb": te, "text_mask": tm,
                                    "ge": self.ref["ge"], "ge512": self.ref["ge512"], "noise": noise})["z"][:, :, :f]
        return self.generate(z)

    def generate(self, z):
        """HiFiGAN in WIN-frame windows (TCM limit); windows clamped to the sequence edges, centers stitched."""
        gen = lambda zw: self.b.run("vits_gen", {"z": np.ascontiguousarray(zw, np.float32), "ge": self.ref["ge"]})["audio"][0, 0]  # noqa: E731
        fz = z.shape[2]
        if fz <= WIN:
            return gen(np.pad(z, ((0, 0), (0, 0), (0, WIN - fz))))[: fz * HOP]
        out, s, step = np.zeros(fz * HOP, np.float32), 0, WIN - 2 * OV
        while s < fz:
            n = min(step, fz - s)
            ws = min(max(s - OV, 0), fz - WIN)
            out[s * HOP : (s + n) * HOP] = gen(z[:, :, ws : ws + WIN])[(s - ws) * HOP : (s - ws + n) * HOP]
            s += n
        return out

    def set_voice(self, voice=None):
        voice = voice or self.voice
        if voice not in self.refs:
            raise KeyError(f"unknown voice {voice!r}; available: {list(self.refs)}")
        self.ref = self.refs[voice]

    def segments(self, text, voice=None, interval=0.3, clause_interval=0.15, hf_filter="off", presence_db=0.0,
                 fade=True, **kw):
        """Yield (audio_float32, info) per segment as soon as it is synthesized, followed by a pause:
        `interval` s after a sentence end, `clause_interval` s where a long sentence was cut at a comma."""
        self.set_voice(voice)
        ref_phones, ref_bert = self.ref["phones"], self.ref["bert"]
        for seg in split_text(text):
            t0 = time.perf_counter()
            phones, word2ph, norm = self.frontend(seg)
            t1 = time.perf_counter()
            bert = self.bert(norm, word2ph)
            t2 = time.perf_counter()
            tokens, eos = self.t2s(np.concatenate([ref_phones, phones]), np.concatenate([ref_bert, bert]), **kw)
            t3 = time.perf_counter()
            audio = post_filter(self.vits(tokens, phones), mode=hf_filter, presence_db=presence_db)
            peak = np.abs(audio).max() if len(audio) else 0
            if peak > 1:
                audio = audio / peak
            if fade:
                audio = fade_edges(audio)
            t4 = time.perf_counter()
            core = seg.rstrip("”」』’）)\"'")
            pause = interval if (core and core[-1] in SENT_END) else clause_interval
            info = dict(text=seg, phones=len(phones), tokens=len(tokens), eos=eos, sec=len(audio) / SR,
                        t_frontend=t1 - t0, t_bert=t2 - t1, t_t2s=t3 - t2, t_vits=t4 - t3)
            yield np.concatenate([audio, np.zeros(int(SR * pause), np.float32)]).astype(np.float32), info

    def tts(self, text, voice=None, **kw):
        pieces, info = [], []
        for a, i in self.segments(text, voice, **kw):
            pieces.append(a)
            info.append(i)
        return (np.concatenate(pieces) if pieces else np.zeros(0, np.float32)), info

    def warmup(self):
        """Touch every lazy load once (jieba dictionary, pypinyin/cn2an tables, G2PW adapter, first execution of
        every NPU graph), so the first real request does not pay for it."""
        for v in self.refs:
            self.tts("你好，我们开始吧。", voice=v, hf_filter="notch", presence_db=1.0)  # touch every code path

def save_wav(path, audio, sr=SR):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ort", "qnn"], default="ort")
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref", default=os.path.join(HERE, "assets"), help="ref_<voice>.npz file or a dir of them")
    ap.add_argument("--voice", default=None, help="voice name (ref_<voice>.npz); default: first one found")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--hf-filter", choices=["off", "notch", "lowpass"], default="off",
                    help="notch: remove only the fine-tuning whine tones; lowpass: 12 kHz low-pass (duller)")
    ap.add_argument("--presence-db", type=float, default=0.0, help="high-shelf boost above ~4 kHz (brighter)")
    ap.add_argument("--onnx-dir", default=os.path.join(HERE, "onnx"), help="ONNX graphs for --backend ort")
    ap.add_argument("--models-dir", default=os.path.join(HERE, "models"))
    ap.add_argument("--qnn-lib", default=os.environ.get("QNN_LIB", os.path.join(HERE, "qnn_libs")),
                    help="dir with libQnnHtp.so / libQnnSystem.so (aarch64-oe-linux-gcc11.2 build on QCS8550)")
    ap.add_argument("--no-burst", action="store_true", help="do not vote for maximum HTP clocks")
    ap.add_argument("--gsv-lib", default=os.environ.get("GSV_QNN_LIB"), help="override lib/libgsv_qnn.so")
    a = ap.parse_args()
    man = json.load(open(os.path.join(HERE, "manifest.json")))
    backend = OrtBackend(man, a.onnx_dir) if a.backend == "ort" else QnnBackend(man, a.models_dir, a.qnn_lib, lib_path=a.gsv_lib, burst=not a.no_burst)
    tb = Timed(backend)
    try:
        tts = GPTSoVITS(tb, a.ref, seed=a.seed)
        t = time.perf_counter()
        audio, info = tts.tts(a.text, voice=a.voice, top_k=a.top_k, temperature=a.temperature, hf_filter=a.hf_filter,
                               presence_db=a.presence_db)
        total = time.perf_counter() - t
    finally:
        if hasattr(backend, "close"):
            backend.close()  # release the NPU before exit (see QnnBackend.__init__)
    save_wav(a.out, audio)
    for i in info:
        print(f"  [{i['sec']:.2f}s, {i['tokens']} tok, eos={i['eos']}] {i['text']}  "
              f"(frontend {i['t_frontend']:.2f}s bert {i['t_bert']:.2f}s t2s {i['t_t2s']:.2f}s vits {i['t_vits']:.2f}s)")
    print("graph calls:", {g: f"{tb.calls[g]}x {tb.secs[g] / max(tb.calls[g], 1) * 1000:.1f}ms" for g in tb.calls})
    print(f"wrote {a.out}: {len(audio) / SR:.2f}s audio in {total:.2f}s (RTF {total / max(len(audio) / SR, 1e-9):.2f}, backend={a.backend})")

if __name__ == "__main__":
    main()
