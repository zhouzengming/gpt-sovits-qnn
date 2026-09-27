"""Step 3: CPU-side assets for the device runtime.

  cpu_weights.npz   tables the device runs on the CPU (embeddings, positional encoding, bert_proj, AR predict layer,
                    VITS phone embedding, semantic codebook)
  ref_<voice>.npz   one reference voice, precomputed: prompt semantic tokens, reference phones + BERT features,
                    ge / ge512 (timbre). ge depends on the SoVITS weights: regenerate after changing them.
  bert_tokenizer/   tokenizer of chinese-roberta-wwm-ext-large (the weights are not needed on the device)

  GSV_ROOT=/path/to/GPT-SoVITS python export/prepare_assets.py --exp logs/myvoice [...] --out work/myvoice/assets
The reference voice is a training clip of the experiment (3~10 s, closest to 6 s; choose one with --ref <clip.wav>)
or any audio with --ref-wav/--ref-text. More voices: run again with another reference and --voice name.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def export_weights(a, out):
    _, sd = C.load_t2s(a.gpt_ckpt)
    m, _ = C.load_vits(a.sovits_pth)
    from AR.modules.embedding import SinePositionalEmbedding

    pe = SinePositionalEmbedding(sd["model.ar_text_embedding.word_embeddings.weight"].shape[1], dropout=0,
                                 scale=False, alpha=True).pe[0, :4000]  # the table used by GPT-SoVITS itself
    f = lambda t: t.detach().float().numpy()  # noqa: E731
    w = dict(text_emb=f(sd["model.ar_text_embedding.word_embeddings.weight"]), bert_proj_w=f(sd["model.bert_proj.weight"]),
             bert_proj_b=f(sd["model.bert_proj.bias"]), text_alpha=f(sd["model.ar_text_position.alpha"]),
             audio_emb=f(sd["model.ar_audio_embedding.word_embeddings.weight"]),
             audio_alpha=f(sd["model.ar_audio_position.alpha"]), pe=f(pe), pred_w=f(sd["model.ar_predict_layer.weight"]),
             vits_text_emb=f(m.enc_p.text_embedding.weight), codebook=f(m.quantizer.vq.layers[0]._codebook.embed))
    codes = torch.randint(0, w["codebook"].shape[0], (1, 1, 7))
    with torch.no_grad():
        assert np.allclose(w["codebook"][codes[0, 0].numpy()].T, m.quantizer.decode(codes)[0].numpy(), atol=1e-6)
    np.savez(os.path.join(out, "cpu_weights.npz"), **w)
    print("[assets] cpu_weights.npz", {k: v.shape for k, v in w.items()})


def prepare_ref(a, root, out):
    import librosa
    import torchaudio
    from feature_extractor import cnhubert
    from module.mel_processing import spectrogram_torch

    m, hd = C.load_vits(a.sovits_pth)
    sr = hd["sampling_rate"]
    pre = root / "GPT_SoVITS" / "pretrained_models"
    # prompt semantic tokens (as TTS._set_prompt_semantic: 16 kHz audio + int(sr * 0.3) zero samples)
    wav16k, _ = librosa.load(a.ref_wav, sr=16000)
    if not 48000 <= len(wav16k) <= 160000:
        raise SystemExit("the reference audio must be 3~10 s long")
    wav16k = np.concatenate([wav16k, np.zeros(int(sr * 0.3), np.float32)])
    cnhubert.cnhubert_base_path = str(pre / "chinese-hubert-base")
    with torch.no_grad():
        feat = cnhubert.get_model().model(torch.from_numpy(wav16k)[None])["last_hidden_state"].transpose(1, 2)
        prompt = m.extract_latent(feat)[0, 0].numpy().astype(np.int64)
    # timbre: spectrogram of the sr audio + speaker-verification embedding of its 16 kHz version (TTS._get_ref_spec)
    raw, raw_sr = torchaudio.load(a.ref_wav)
    raw = raw.mean(0, keepdim=True) if raw.shape[0] == 2 else raw
    audio = torchaudio.functional.resample(raw, raw_sr, sr) if raw_sr != sr else raw
    if audio.abs().max() > 1:
        audio /= min(2, audio.abs().max())
    spec = spectrogram_torch(audio, hd["filter_length"], sr, hd["hop_length"], hd["win_length"], center=False)
    with C.cwd(root):  # sv.py and the text frontend resolve their model paths relative to the GPT-SoVITS root
        from sv import SV
        from text import cleaned_text_to_sequence
        from text.cleaner import clean_text

        sv_emb = SV("cpu", False).compute_embedding3(torchaudio.functional.resample(audio, sr, 16000))
        text = a.ref_text.strip()
        if text[-1] not in "，。？！,.?!~…、；;：:":
            text += "。"
        phones, word2ph, norm = clean_text(text, "zh", "v2")
    with torch.no_grad():
        rm = torch.ones_like(spec[:1, :1, :])
        ge = m.prelu(m.ref_enc(spec[:, :704] * rm, rm) + m.sv_emb(sv_emb).unsqueeze(-1))
        ge512 = m.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    bdir = pre / "chinese-roberta-wwm-ext-large"
    tok, bert = AutoTokenizer.from_pretrained(bdir), AutoModelForMaskedLM.from_pretrained(bdir).float().eval()
    with torch.no_grad():
        h = bert(**tok(norm, return_tensors="pt"), output_hidden_states=True)["hidden_states"][-3][0, 1:-1]
    assert len(word2ph) == len(norm)
    bert_feat = torch.cat([h[i].repeat(word2ph[i], 1) for i in range(len(word2ph))]).numpy().astype(np.float32)
    ids = np.array(cleaned_text_to_sequence(phones, "v2"), np.int64)
    np.savez(os.path.join(out, f"ref_{a.voice}.npz"), prompt=prompt, phones=ids, bert=bert_feat,
             ge=ge.numpy().astype(np.float32), ge512=ge512.numpy().astype(np.float32), text=np.array(norm))
    print(f"[assets] ref_{a.voice}.npz: {len(prompt)} prompt tokens, {len(ids)} phones, text={norm}")
    if len(prompt) + len(ids) > C.P - 60:
        print(f"[assets] warning: the reference takes {len(prompt) + len(ids)} of the {C.P} prefill positions; "
              "use a shorter reference audio/text")
    tdir = os.path.join(out, "bert_tokenizer")
    os.makedirs(tdir, exist_ok=True)
    for f in ("config.json", "tokenizer.json"):
        shutil.copy(bdir / f, os.path.join(tdir, f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_model_args(ap)
    ap.add_argument("--ref", help="training clip of --exp to use as reference (name in 5-wav32k/)")
    ap.add_argument("--ref-wav", help="any 3~10 s reference audio instead (needs --ref-text)")
    ap.add_argument("--ref-text", help="transcript of --ref-wav (Chinese)")
    ap.add_argument("--voice", help="voice name (request field `voice` of the TTS service); default: experiment name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = C.resolve_models(a)
    if not a.ref_wav:
        if not a.exp:
            raise SystemExit("give --ref-wav/--ref-text, or --exp to pick a training clip")
        a.ref_wav, a.ref_text, dur = C.pick_reference(a.exp, a.ref)
        print(f"[assets] reference: {os.path.basename(a.ref_wav)} ({dur:.1f} s) {a.ref_text}")
    elif not a.ref_text:
        raise SystemExit("--ref-wav needs --ref-text")
    a.voice = a.voice or (a.exp_info["name"] if a.exp_info else "default")
    a.ref_wav, out = os.path.abspath(a.ref_wav), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    export_weights(a, out)
    prepare_ref(a, root, out)


if __name__ == "__main__":
    main()
