"""Optional: check the compiled context binaries on a real device through AI Hub inference jobs.

Runs a synthesis locally (onnxruntime on the same graphs, fp32) recording real inputs of every graph, sends them
to the device and compares the outputs (fp16 on the NPU) with the fp32 ones. The package must have been built with
`make_deploy.py --with-onnx`.

  python export/aihub_verify.py --deploy dist/myvoice [--device "QCS8550 (Proxy)"]
"""
import argparse
import json
import os
import sys

import numpy as np
import qai_hub as hub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aihub_compile import call, wait  # noqa: E402

TEXT = "今天天气不错，我们一起去公园散步吧！路上可以看到很多盛开的花，还有在湖边悠闲游泳的小鸭子。你觉得怎么样？"
KEEP = {"t2s_decode": 4, "vits_gen": 4}  # samples per graph (decode inputs are ~100 MB each)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", required=True)
    ap.add_argument("--device", default="QCS8550 (Proxy)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.deploy))
    import gsv_runtime as R

    man = json.load(open(os.path.join(a.deploy, "manifest.json")))
    ort = R.OrtBackend(man, os.path.join(a.deploy, "onnx"))

    class Rec:  # records the first samples of every graph (copies: the runtime mutates caches in place)
        def __init__(self, manifest):
            self.full_manifest, self.man = manifest, R.graphs_of(manifest)
            self.calls, self.seen = {g: [] for g in self.man}, {g: 0 for g in self.man}

        def run(self, g, feeds):
            out = ort.run(g, feeds)
            self.seen[g] += 1
            stride = 40 if g == "t2s_decode" else 1
            if len(self.calls[g]) < KEEP.get(g, 3) and (self.seen[g] - 1) % stride == 0:
                self.calls[g].append(({k: v.copy() for k, v in feeds.items()}, {k: v.copy() for k, v in out.items()}))
            return out
    rec = Rec(man)
    tts = R.GPTSoVITS(rec, os.path.join(a.deploy, "assets"), seed=0)
    tts.tts(TEXT)
    W = tts.w["pred_w"]
    dev = hub.Device(a.device)
    ok = True
    for g, samples in rec.calls.items():
        ctx = call(lambda: hub.get_model(man[g]["ctx_model"]), "get model")
        ins = {n: [f[n] for f, _ in samples] for n in samples[0][0]}
        job = call(lambda: hub.submit_inference_job(model=ctx, device=dev, inputs=ins, name=f"gsv-verify-{g}"),
                   f"submit {g}", timeout=1800, tries=5)
        wait(job.job_id, f"{g} inference")
        d = call(lambda: hub.get_job(job.job_id).download_output_data(), "download", timeout=1800)
        names = [o[0] for o in man[g]["outputs"]]
        for s, (f, ref) in enumerate(samples):
            o = {names[i]: d[f"output_{i}"][s] for i in range(len(names))}
            if g in ("t2s_decode", "t2s_prefill"):
                if g == "t2s_prefill":
                    n = int((f["mask"][0, 0, :, 0] == 0).sum())
                    x, y = o["h"][0, n - 1] @ W.T, ref["h"][0, n - 1] @ W.T
                else:
                    x, y = o["logits"][0], ref["logits"][0]
                m = len(set(np.argsort(-x)[:15]) & set(np.argsort(-y)[:15])) / 15
                msg, good = f"top-15 overlap {m:.2f}", m >= 0.9
            else:
                k = names[0]
                x, y = o[k].astype(np.float64).ravel(), ref[k].astype(np.float64).ravel()
                cos = float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
                msg, good = f"cos {cos:.5f}", cos > 0.995
            ok &= good
            print(f"[verify] {g} sample {s}: {msg} {'ok' if good else 'MISMATCH'}")
    print("[verify]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
