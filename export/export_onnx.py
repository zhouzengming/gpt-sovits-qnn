"""Step 1: GPT-SoVITS v2Pro -> static-shape ONNX graphs for the Qualcomm HTP (NPU).

    g2pw        : polyphone BERT body -> hidden state of the queried char        [B chars x S tokens]
    bert        : chinese-roberta-wwm-ext-large, first 22 layers (= hidden_states[-3] used by GPT-SoVITS)
    t2s_prefill : GPT, all layers over [phones + reference semantic tokens]     -> hidden, k, v
    t2s_decode  : GPT, one token against a fixed-size kv cache (mask marks the valid slots)
    vits_enc    : SoVITS text/semantic encoder + flow                          -> latent z
    vits_gen    : HiFiGAN generator on a WIN-frame window (run in overlapping windows by the runtime)

Embedding lookups, positional encoding, sampling, the kv-cache bookkeeping and the G2PW output head run on the
CPU (see device/gsv_runtime.py). Output: <out>/<graph>.onnx/{model.onnx, model.data}, <out>/g2pw_head/*.npy and
<out>/export.json.

  GSV_ROOT=/path/to/GPT-SoVITS python export/export_onnx.py --exp logs/myvoice [--gpt-epoch 15 --sovits-epoch 16] \\
      --out work/myvoice/onnx
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


# ------------------------------------------------------------------ GPT (text-to-semantic)
class T2SLayer(nn.Module):
    """Post-norm transformer layer of GPT-SoVITS' AR model, with explicit prefill / single-step decode paths."""

    def __init__(self, sd, i, d, heads):
        super().__init__()
        p = f"model.h.layers.{i}."
        self.heads, self.dh = heads, d // heads
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.l1, self.l2 = nn.Linear(d, sd[p + "linear1.weight"].shape[0]), nn.Linear(sd[p + "linear1.weight"].shape[0], d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv.weight.data, self.qkv.bias.data = sd[p + "self_attn.in_proj_weight"], sd[p + "self_attn.in_proj_bias"]
        self.out.weight.data, self.out.bias.data = sd[p + "self_attn.out_proj.weight"], sd[p + "self_attn.out_proj.bias"]
        for n, m in (("linear1", self.l1), ("linear2", self.l2), ("norm1", self.n1), ("norm2", self.n2)):
            m.weight.data, m.bias.data = sd[p + n + ".weight"], sd[p + n + ".bias"]

    def post(self, x, a):
        x = self.n1(x + self.out(a))
        return self.n2(x + self.l2(F.relu(self.l1(x))))

    def prefill(self, x, mask):  # x [1,P,d], additive mask [1,1,P,P]
        n = x.shape[1]
        q, k, v = (t.view(1, n, self.heads, self.dh).transpose(1, 2) for t in self.qkv(x).chunk(3, -1))
        s = torch.matmul(q, k.transpose(2, 3)) * (1.0 / math.sqrt(self.dh)) + mask
        a = torch.matmul(torch.softmax(s, -1), v).transpose(1, 2).reshape(1, n, -1)
        return self.post(x, a), k[0], v[0]

    def decode(self, x, kT, vc, mask):  # x [1,1,d], kT [H,dh,L], vc [H,L,dh], mask [1,1,1,L]
        q, k, v = (t.view(self.heads, 1, self.dh) for t in self.qkv(x).chunk(3, -1))
        q = q * (1.0 / math.sqrt(self.dh))
        w = torch.softmax(torch.cat([torch.matmul(q, kT) + mask[0], (q * k).sum(-1, keepdim=True)], -1), -1)
        a = (torch.matmul(w[..., :-1], vc) + w[..., -1:] * v).transpose(0, 1).reshape(1, 1, -1)
        return self.post(x, a), k.transpose(1, 2), v


class T2S(nn.Module):
    def __init__(self, sd, cfg):
        super().__init__()
        m = cfg["model"]
        self.d, self.heads, self.n = m["hidden_dim"], m["head"], m["n_layer"]
        self.layers = nn.ModuleList([T2SLayer(sd, i, self.d, self.heads) for i in range(self.n)])
        self.pred = nn.Linear(self.d, sd["model.ar_predict_layer.weight"].shape[0], bias=False)
        self.pred.weight.data = sd["model.ar_predict_layer.weight"]


class Prefill(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, xy, mask):
        ks, vs = [], []
        for layer in self.t.layers:
            xy, k, v = layer.prefill(xy, mask)
            ks.append(k)
            vs.append(v)
        return xy, torch.stack(ks), torch.stack(vs)


class Decode(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x, kT_cache, v_cache, mask):
        ks, vs = [], []
        for i, layer in enumerate(self.t.layers):
            x, k, v = layer.decode(x, kT_cache[i], v_cache[i], mask)
            ks.append(k)
            vs.append(v)
        return self.t.pred(x[:, 0]), torch.stack(ks), torch.stack(vs)


# ------------------------------------------------------------------ SoVITS
class VitsEnc(nn.Module):
    """enc_p (with explicit padding masks) + reverse flow -> z. ge/ge512 (timbre) are precomputed per voice."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ssl, y_mask, text_emb, text_mask, ge512, ge, noise):
        e = self.m.enc_p
        y = e.ssl_proj(ssl * y_mask) * y_mask
        y = e.encoder_ssl(y * y_mask, y_mask)
        t = e.encoder_text(text_emb * text_mask, text_mask)
        y = e.mrte(y, y_mask, t, text_mask, ge512)
        y = e.encoder2(y * y_mask, y_mask)
        m_p, logs_p = torch.split(e.proj(y) * y_mask, e.out_channels, dim=1)
        return self.m.flow(m_p + noise * torch.exp(logs_p), y_mask, g=ge, reverse=True) * y_mask


class VitsGen(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.dec = m.dec

    def forward(self, z, ge):
        return self.dec(z, g=ge)


# ------------------------------------------------------------------ BERT
def load_bert(bert_dir):
    from transformers import AutoConfig, AutoModelForMaskedLM

    m = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert_dir, attn_implementation="eager"))
    m.load_state_dict(torch.load(os.path.join(bert_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True),
                      strict=False)
    return m.float().eval()


class BertTrunc(nn.Module):
    def __init__(self, m, n_layers):
        super().__init__()
        self.emb, self.layers = m.bert.embeddings, m.bert.encoder.layer[:n_layers]

    def forward(self, input_ids, mask_add):  # mask_add [1,1,1,S]: 0 valid / negative for padding
        x = self.emb(input_ids=input_ids, token_type_ids=torch.zeros_like(input_ids))
        for layer in self.layers:
            x = layer(x, attention_mask=mask_add)[0]
        return x


# ------------------------------------------------------------------ G2PW (from the released g2pW.onnx)
def export_g2pw(src, out_dir, head_dir):
    """Static B x S, int32 inputs, clamped indices, graph cut at the hidden state of the query char.
    The head runs in numpy fp32 on the CPU: in fp16 its masked softmax (exp(logit - global max) of the allowed
    readings) underflows and the POS ArgMax flips on near-ties - ~4% of the predictions changed on the device.
    Index inputs are clamped because out-of-range indices (AI Hub profiles with random inputs) crash the HTP."""
    import onnx
    import onnxslim
    from onnx import TensorProto, helper, numpy_helper

    B, S = C.G2PW_B, C.G2PW_S
    m = onnx.load(src)
    g = m.graph
    shapes = {"input_ids": [B, S], "token_type_ids": [B, S], "attention_mask": [B, S], "phoneme_mask": [B, None],
              "char_ids": [B], "position_ids": [B]}
    init = {i.name: i for i in g.initializer}
    n_vocab = numpy_helper.to_array(init["bert.embeddings.word_embeddings.weight"]).shape[0]
    n_type = numpy_helper.to_array(init["bert.embeddings.token_type_embeddings.weight"]).shape[0]
    n_char = numpy_helper.to_array(init["char_descriptor.weight"]).shape[0]
    clamp = {"input_ids": n_vocab - 1, "token_type_ids": n_type - 1, "char_ids": n_char - 1, "position_ids": S - 1}
    new = []
    for inp in g.input:
        dims = inp.type.tensor_type.shape.dim
        for d, v in zip(dims, shapes[inp.name]):
            if v is not None:
                d.ClearField("dim_param")
                d.dim_value = v
        if inp.type.tensor_type.elem_type == TensorProto.INT64:
            old, i64 = inp.name, inp.name + "_i64"
            for n in g.node:
                for j, x in enumerate(n.input):
                    if x == old:
                        n.input[j] = i64
            inp.type.tensor_type.elem_type = TensorProto.INT32
            if old in clamp:
                g.initializer.extend([helper.make_tensor(old + "_lo", TensorProto.INT64, [], [0]),
                                      helper.make_tensor(old + "_hi", TensorProto.INT64, [], [clamp[old]])])
                new += [helper.make_node("Cast", [old], [old + "_raw"], to=TensorProto.INT64),
                        helper.make_node("Clip", [old + "_raw", old + "_lo", old + "_hi"], [i64])]
            else:
                new.append(helper.make_node("Cast", [old], [i64], to=TensorProto.INT64))
    for k, nd in enumerate(new):
        g.node.insert(k, nd)
    del g.value_info[:]
    for o in g.output:
        for d, v in zip(o.type.tensor_type.shape.dim, [B]):
            d.ClearField("dim_param")
            d.dim_value = v
    m = onnxslim.slim(m)  # static shapes -> shape subgraphs fold, the descriptor bias becomes a constant
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    cls = next(n for n in g.node if n.op_type == "Gemm" and "classifier.weight" in n.input)
    hidden = cls.input[0]
    # descriptor mask = sigmoid(bias + char_descriptor[char] + second_order_descriptor[char * n_pos + pos])
    add = next(n for n in g.node if n.op_type == "Add" and any(
        p.op_type == "Gather" and "char_descriptor.weight" in p.input for p in g.node if p.output[0] in n.input))
    bias = next(init[x] for x in add.input if x in init)
    bias = bias.reshape(-1, bias.shape[-1])
    assert np.allclose(bias, bias[:1]), "unexpected G2PW head structure (descriptor bias differs per row)"
    os.makedirs(head_dir, exist_ok=True)
    for k, v in {"pos_w": "pos_classifier.weight", "pos_b": "pos_classifier.bias", "cls_w": "classifier.weight",
                 "cls_b": "classifier.bias", "char_desc": "char_descriptor.weight",
                 "second_order": "second_order_descriptor.weight"}.items():
        np.save(os.path.join(head_dir, k + ".npy"), init[v].astype(np.float32))
    np.save(os.path.join(head_dir, "desc_bias.npy"), bias[0].astype(np.float32))
    del g.output[:]
    g.output.extend([helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [B, init["classifier.weight"].shape[1]])])
    g.node.append(helper.make_node("Identity", [hidden], ["hidden"], name="hidden_out"))
    m = onnxslim.slim(m)  # drops the head and its tables
    C.save_onnx_dir(m, out_dir)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_model_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=list(C.GRAPHS), choices=list(C.GRAPHS))
    a = ap.parse_args()
    root = C.resolve_models(a)
    pre = root / "GPT_SoVITS" / "pretrained_models"
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    cfg, sd = C.load_t2s(a.gpt_ckpt)
    t2s = T2S(sd, cfg).eval()
    vits, hd = C.load_vits(a.sovits_pth)
    d, H, n = t2s.d, t2s.heads, t2s.n
    for g in a.only:
        path = os.path.join(out, C.GRAPHS[g] + ".onnx")
        print(f"[export] {g} -> {path}", flush=True)
        if g == "t2s_prefill":
            C.export_torch(Prefill(t2s), (torch.randn(1, C.P, d), torch.zeros(1, 1, C.P, C.P)), ["xy", "mask"],
                           ["h", "k", "v"], path)
        elif g == "t2s_decode":
            C.export_torch(Decode(t2s), (torch.randn(1, 1, d), torch.randn(n, H, d // H, C.L),
                                         torch.randn(n, H, C.L, d // H), torch.zeros(1, 1, 1, C.L)),
                           ["x", "kT_cache", "v_cache", "mask"], ["logits", "k_new", "v_new"], path)
        elif g == "vits_enc":
            C.export_torch(VitsEnc(vits), (torch.randn(1, 768, C.FR), torch.ones(1, 1, C.FR), torch.randn(1, 192, C.TT),
                                           torch.ones(1, 1, C.TT), torch.randn(1, 512, 1), torch.randn(1, 1024, 1),
                                           torch.zeros(1, 192, C.FR)),
                           ["ssl", "y_mask", "text_emb", "text_mask", "ge512", "ge", "noise"], ["z"], path)
        elif g == "vits_gen":
            C.export_torch(VitsGen(vits), (torch.randn(1, 192, C.WIN), torch.randn(1, 1024, 1)), ["z", "ge"], ["audio"], path)
        elif g == "bert":
            ids = torch.zeros(1, C.BERT_S, dtype=torch.int32)
            C.export_torch(BertTrunc(load_bert(pre / "chinese-roberta-wwm-ext-large"), C.BERT_LAYERS),
                           (ids, torch.zeros(1, 1, 1, C.BERT_S)), ["input_ids", "mask_add"], ["h"], path, slim=False)
        elif g == "g2pw":
            export_g2pw(str(root / "GPT_SoVITS" / "text" / "G2PWModel" / "g2pW.onnx"), path, os.path.join(out, "g2pw_head"))
    info = {"graphs": C.GRAPHS, "t2s": {"layers": n, "heads": H, "dim": d, "vocab": int(t2s.pred.weight.shape[0])},
            "sampling_rate": hd["sampling_rate"], "hop_length": hd["hop_length"],
            "shapes": {"P": C.P, "L": C.L, "FR": C.FR, "TT": C.TT, "WIN": C.WIN, "OV": C.OV, "BERT_S": C.BERT_S,
                       "G2PW_B": C.G2PW_B, "G2PW_S": C.G2PW_S},
            "gpt_ckpt": a.gpt_ckpt, "sovits_pth": a.sovits_pth,
            "experiment": {k: a.exp_info[k] for k in ("name", "gpt_epoch", "sovits_epoch")} if a.exp_info else None}
    json.dump(info, open(os.path.join(out, "export.json"), "w"), indent=1, ensure_ascii=False)
    print("[export] done:", out)


if __name__ == "__main__":
    main()
