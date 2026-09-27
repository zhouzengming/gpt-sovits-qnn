"""Optional: rank the SoVITS epochs of a training folder by two artifacts that show up with fine-tuning on small
data sets (measured on a 19-minute voice: a late epoch buzzed on every breath, earlier ones did not).

  breath tonality  narrowband (tonal) energy inside the breaths of the resynthesized training clips. A breath is
                   broadband noise; tonal peaks in it are heard as a buzz / hum on inhalations. Sum over 400-4000 Hz
                   of (spectrum - local median - 6 dB)+ on the ground-truth breath windows.
  whine            excess of the 500 Hz-multiple tonal comb (11 / 13 / 13.5 / 14.5 / 15.7 kHz) that the HiFiGAN
                   upsampling develops during fine-tuning (the pretrained model has none).

Every G_<step>.pth of logs/<exp>/logs_s2_*/ resynthesizes the same training clips from their own semantic tokens
(6-name2semantic.tsv) and phones (2-name2text.txt); outputs are time-aligned with the recordings (5-wav32k).
Lower is better for both; listen to the candidates before choosing.

  GSV_ROOT=/path/to/GPT-SoVITS python export/eval_sovits_ckpts.py --exp logs/myvoice [--clips 24]
"""
import argparse
import os
import sys

import numpy as np
import scipy.signal as ss
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SR, HOP = 32000, 320
COMB_HZ = (11000, 13000, 13500, 14500, 15700)


def breath_windows(y):
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    _, voiced, _ = librosa.pyin(y, fmin=70, fmax=600, sr=SR, frame_length=2048, hop_length=HOP)
    db = 20 * np.log10(rms + 1e-6)
    m = np.append((~voiced) & (db > -50) & (db < -22), False)  # unvoiced, quiet but not silent
    out, s = [], None
    for i, v in enumerate(m):
        if v and s is None:
            s = i
        if not v and s is not None:
            if i - s >= 10:
                out.append((s * HOP, i * HOP))
            s = None
    return out


def tonality(y, windows):
    ps = [ss.welch(y[a:b], fs=SR, nperseg=2048, noverlap=1536)[1] for a, b in windows if b - a >= 2048]
    if not ps:
        return np.nan
    f = ss.welch(np.zeros(4096), fs=SR, nperseg=2048)[0]
    s = 10 * np.log10(np.mean(ps, 0) + 1e-20)
    ex = s - ss.medfilt(s, 31)
    return float(np.clip(ex[(f > 400) & (f < 4000)] - 6, 0, None).sum())


def whine(y):
    f, p = ss.welch(y, fs=SR, nperseg=8192)
    d = 10 * np.log10(p + 1e-20)
    ex = d - ss.medfilt(d, 101)
    return float(max(ex[np.argmin(abs(f - hz))] for hz in COMB_HZ))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gsv-root", default=os.environ.get("GSV_ROOT"))
    ap.add_argument("--exp", required=True)
    ap.add_argument("--clips", type=int, default=24, help="training clips with breaths to resynthesize")
    a = ap.parse_args()
    import glob

    import librosa

    C.setup_gsv(a.gsv_root)
    from module.mel_processing import spectrogram_torch
    from text import cleaned_text_to_sequence

    exp = C.experiment(a.exp)
    sem = {l.split("\t")[0]: l.split("\t")[1] for l in open(os.path.join(a.exp, "6-name2semantic.tsv")).read().splitlines()[1:]}
    ph = {l.split("\t")[0]: l.split("\t")[1] for l in open(os.path.join(a.exp, "2-name2text.txt"), encoding="utf-8").read().splitlines()}
    items = []
    for p in sorted(glob.glob(os.path.join(a.exp, "5-wav32k", "*.wav"))):
        n = os.path.basename(p)
        sv = os.path.join(a.exp, "7-sv_cn", n + ".pt")
        if n not in sem or n not in ph or not os.path.exists(sv):
            continue
        y, _ = librosa.load(p, sr=SR)
        w = breath_windows(y)
        if w:
            items.append((n, y, w, torch.load(sv, map_location="cpu").float().reshape(1, -1)))
        if len(items) == a.clips:
            break
    print(f"{len(items)} clips, {sum(len(i[2]) for i in items)} breath windows; ground truth tonality "
          f"{np.nanmean([tonality(y, w) for _, y, w, _ in items]):.2f}")
    print(f"{'epoch':>6} {'breath tonality':>16} {'clips > 3':>10} {'whine dB':>9}")
    for ep in exp["sovits_epochs"]:
        m, hd = C.load_vits(exp["sovits_files"][ep])
        ts, wh = [], []
        for n, y, w, sv in items:
            refer = spectrogram_torch(torch.from_numpy(y)[None], hd["filter_length"], SR, hd["hop_length"],
                                      hd["win_length"], center=False)
            torch.manual_seed(0)
            with torch.no_grad():
                g = m(torch.LongTensor([[list(map(int, sem[n].split()))]]),
                      torch.LongTensor([cleaned_text_to_sequence(ph[n].split(), "v2")]), refer, noise_scale=0.5,
                      sv_emb=sv)[0, 0].numpy()
            ts.append(tonality(g, [(s, min(e, len(g))) for s, e in w]))
            wh.append(whine(g))
        print(f"{'e' + str(ep):>6} {np.nanmean(ts):16.2f} {sum(t > 3 for t in ts):>6}/{len(ts):<3} {np.mean(wh):9.1f}", flush=True)


if __name__ == "__main__":
    main()
