"""Step 2: check every exported graph (onnxruntime, fp32, CPU) against the original GPT-SoVITS implementation.

  GSV_ROOT=/path/to/GPT-SoVITS python export/verify_onnx.py --exp logs/myvoice [...] --onnx-dir work/myvoice/onnx
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

CORPUS = ["我们一起去银行，把重要的东西还给他。", "这个行业的发展很快，他还在长大。", "长江的长度是多少？重庆的重要景点有哪些？",
          "他背着背包，在音乐会上唱了一首好听的歌。", "朝阳的朝气蓬勃，朝代更替。", "为了调查这件事，他调整了计划。",
          "都市里的人都很忙，还得还钱。", "差不多了，参差不齐的差距还是很大。", "他的血液循环不好，血压有点高。",
          "这首曲子很好听，别把铁丝弄弯曲了。", "请把门关好，着急也没用，他着了凉。", "我觉得睡了一觉以后好多了。",
          "今天天气不错，我们一起去公园散步吧！", "路上可以看到很多盛开的花，还有在湖边悠闲游泳的小鸭子。",
          "他们的数量很多，数一数就知道了。", "这种药的作用很强，要按量服用。", "那里的空气很好，只是交通不太便利。",
          "他正在给大家讲解这道难题的解法。", "我们要处理好这些事情，不能处处都靠别人。"]
so = ort.SessionOptions()
so.log_severity_level = 3


def sess(onnx_dir, g):
    return ort.InferenceSession(os.path.join(onnx_dir, C.GRAPHS[g] + ".onnx", "model.onnx"), so,
                                providers=["CPUExecutionProvider"])


def rel(a, b):
    return float(np.abs(np.asarray(a, np.float64) - b).max() / (np.abs(b).max() + 1e-12))


def check_t2s(a, steps=30):
    from AR.models.t2s_model import Text2SemanticDecoder

    cfg, sd = C.load_t2s(a.gpt_ckpt)
    ref = Text2SemanticDecoder(cfg, top_k=3)
    ref.load_state_dict({k[len("model."):]: v for k, v in sd.items()}, strict=False)
    ref.eval()
    n_layer, d = cfg["model"]["n_layer"], cfg["model"]["hidden_dim"]
    H = cfg["model"]["head"]
    x_len, y_len, n = 60, 100, 160
    x = torch.randint(0, cfg["model"]["phoneme_vocab_size"], (1, x_len))
    bert, y = torch.randn(1, 1024, x_len), torch.randint(0, 1024, (1, y_len))
    toks = torch.randint(0, 1024, (steps,))
    with torch.no_grad():
        xe = ref.ar_text_position(ref.ar_text_embedding(x) + ref.bert_proj(bert.transpose(1, 2)))
        xy = torch.cat([xe, ref.ar_audio_position(ref.ar_audio_embedding(y))], 1)
        m = torch.zeros(n, n, dtype=torch.bool)
        m[:x_len, x_len:] = True
        m[x_len:, x_len:] = torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), 1)
        h, kc, vc = ref.t2s_transformer.process_prompt(xy, m[None, None].expand(1, H, n, n), None)
        ref_logits = [ref.ar_predict_layer(h[:, -1])]
        embs = []
        for i in range(steps):
            e = ref.ar_audio_embedding(toks[i:i + 1][None]) + ref.ar_audio_position.alpha * ref.ar_audio_position.pe[:, y_len + i]
            embs.append(e.numpy().astype(np.float32))
            h, kc, vc = ref.t2s_transformer.decode_next_token(e, kc, vc)
            ref_logits.append(ref.ar_predict_layer(h[:, -1]))
    sp, sdc = sess(a.onnx_dir, "t2s_prefill"), sess(a.onnx_dir, "t2s_decode")
    xy_p = np.zeros((1, C.P, d), np.float32)
    xy_p[:, :n] = xy.numpy()
    mask = np.full((1, 1, C.P, C.P), -100.0, np.float32)
    mask[0, 0, :n, :n] = np.where(m.numpy(), -100.0, 0.0)
    hs, k, v = sp.run(None, {"xy": xy_p, "mask": mask})
    out = [hs[:, n - 1] @ sd["model.ar_predict_layer.weight"].numpy().T]
    kT = np.zeros((n_layer, H, d // H, C.L), np.float32)
    vC = np.zeros((n_layer, H, C.L, d // H), np.float32)
    kT[..., :n], vC[:, :, :n] = k[:, :, :n].transpose(0, 1, 3, 2), v[:, :, :n]
    for i in range(steps):
        dm = np.full((1, 1, 1, C.L), -100.0, np.float32)
        dm[..., :n + i] = 0
        lg, kn, vn = sdc.run(None, {"x": embs[i], "kT_cache": kT, "v_cache": vC, "mask": dm})
        kT[..., n + i:n + i + 1], vC[:, :, n + i:n + i + 1] = kn, vn
        out.append(lg)
    r, o = torch.cat(ref_logits).numpy(), np.concatenate(out)
    top1 = (r.argmax(-1) == o.argmax(-1)).mean()
    print(f"[t2s]  prefill + {steps} decode steps: logits rel err {rel(o, r):.2e}, top-1 agreement {top1:.3f}")
    return top1 == 1.0 and rel(o, r) < 1e-3


def check_vits(a):
    from module.mel_processing import spectrogram_torch

    m, hd = C.load_vits(a.sovits_pth)
    torch.manual_seed(0)
    t_sem, t_txt = 150, 60
    codes = torch.randint(0, 1024, (1, 1, t_sem))
    text = torch.randint(0, m.enc_p.text_embedding.num_embeddings, (1, t_txt))
    refer = spectrogram_torch(torch.randn(1, 32000 * 5) * 0.1, hd["filter_length"], hd["sampling_rate"],
                              hd["hop_length"], hd["win_length"], center=False)
    sv = torch.randn(1, 20480)
    with torch.no_grad():
        rm = torch.ones_like(refer[:1, :1, :])
        ge = m.prelu(m.ref_enc(refer[:, :704] * rm, rm) + m.sv_emb(sv).unsqueeze(-1))
        ge512 = m.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
        q = torch.repeat_interleave(m.quantizer.decode(codes), 2, dim=2)
        te = m.enc_p.text_embedding(text).transpose(1, 2)
        noise = torch.randn(1, 192, q.shape[2]) * 0.5
        _, m_p, logs_p, y_mask = m.enc_p(q, text, ge512)
        z_ref = m.flow(m_p + noise * torch.exp(logs_p), y_mask, g=ge, reverse=True)
        ref = m.dec(z_ref * y_mask, g=ge)[0, 0].numpy()
    f, nt = q.shape[2], t_txt
    pad = lambda x, n: np.pad(x.numpy().astype(np.float32), ((0, 0), (0, 0), (0, n - x.shape[2])))  # noqa: E731
    z = sess(a.onnx_dir, "vits_enc").run(None, {
        "ssl": pad(q, C.FR), "y_mask": pad(torch.ones(1, 1, f), C.FR), "text_emb": pad(te, C.TT),
        "text_mask": pad(torch.ones(1, 1, nt), C.TT), "ge512": ge512.numpy(), "ge": ge.numpy(),
        "noise": pad(noise, C.FR)})[0][:, :, :f]
    gen = sess(a.onnx_dir, "vits_gen")
    hop = hd["hop_length"]
    out, s, step = np.zeros(f * hop, np.float32), 0, C.WIN - 2 * C.OV
    while s < f:  # same window schedule as device/gsv_runtime.py GPTSoVITS.generate
        k = min(step, f - s)
        ws = min(max(s - C.OV, 0), f - C.WIN)
        w = gen.run(None, {"z": z[:, :, ws:ws + C.WIN].astype(np.float32), "ge": ge.numpy()})[0][0, 0]
        out[s * hop:(s + k) * hop] = w[(s - ws) * hop:(s - ws + k) * hop]
        s += k
    snr = 10 * np.log10((ref ** 2).sum() / ((out - ref) ** 2).sum())
    print(f"[vits] encoder z rel err {rel(z, z_ref.numpy()):.2e}; windowed generator vs full-length decode: SNR {snr:.1f} dB")
    return snr > 60


def check_bert(a, root):
    from transformers import AutoTokenizer
    sys.path.insert(0, HERE)
    from export_onnx import load_bert

    bdir = root / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"
    tok, full = AutoTokenizer.from_pretrained(bdir), load_bert(bdir)
    s = sess(a.onnx_dir, "bert")
    worst = 0.0
    for text in CORPUS[:6]:
        inp = tok(text, return_tensors="pt")
        with torch.no_grad():
            ref = full(**inp, output_hidden_states=True)["hidden_states"][-3][0].numpy()
        n = inp["input_ids"].shape[1]
        ids = np.zeros((1, C.BERT_S), np.int32)
        ids[0, :n] = inp["input_ids"][0].numpy()
        ma = np.full((1, 1, 1, C.BERT_S), -100.0, np.float32)
        ma[..., :n] = 0
        worst = max(worst, rel(s.run(None, {"input_ids": ids, "mask_add": ma})[0][0, :n], ref))
    print(f"[bert] hidden_states[-3] rel err (worst of 6 sentences) {worst:.2e}")
    return worst < 1e-3


def check_g2pw(a, root):
    from text.g2pw.onnx_api import G2PWOnnxConverter  # GPT-SoVITS' original (dynamic, CPU) G2PW

    conv = G2PWOnnxConverter(model_dir=str(root / "GPT_SoVITS" / "text" / "G2PWModel"), style="pinyin",
                             model_source=str(root / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"),
                             enable_non_tradional_chinese=True)
    calls, real = [], conv.session_g2pW.run

    def rec(names, feeds):
        out = real(names, feeds)
        calls.append(({k: v.copy() for k, v in feeds.items()}, out[0].copy()))
        return out
    conv.session_g2pW.run = rec
    for s in CORPUS:
        conv(s)
    sys.path.insert(0, os.path.join(HERE, "..", "device"))
    from gsv_runtime import G2pwHead, NpuG2pw

    class OrtG2pw:  # backend with the static graph, as the device runtime calls it
        def __init__(self):
            self.s = sess(a.onnx_dir, "g2pw")

        def run(self, g, feeds):
            return {"hidden": self.s.run(None, feeds)[0]}
    npu = NpuG2pw(OrtG2pw(), head=G2pwHead(os.path.join(a.onnx_dir, "g2pw_head")))
    agree = rows = 0
    for feeds, ref in calls:
        p = npu.run(None, feeds)[0]
        agree += int((p.argmax(1) == ref.argmax(1)).sum())
        rows += len(ref)
    print(f"[g2pw] static graph + CPU head vs original model: argmax agreement {agree}/{rows} ({len(calls)} calls)")
    return agree == rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_model_args(ap)
    ap.add_argument("--onnx-dir", required=True)
    a = ap.parse_args()
    a.onnx_dir = os.path.abspath(a.onnx_dir)
    root = C.resolve_models(a)
    ok = {"t2s": check_t2s(a), "vits": check_vits(a), "bert": check_bert(a, root), "g2pw": check_g2pw(a, root)}
    print("[verify]", "PASS" if all(ok.values()) else f"FAIL {[k for k, v in ok.items() if not v]}")
    sys.exit(0 if all(ok.values()) else 1)


if __name__ == "__main__":
    main()
