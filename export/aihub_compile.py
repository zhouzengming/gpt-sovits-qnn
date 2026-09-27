"""Step 4: compile every graph for the NPU on Qualcomm AI Hub (fp16 QNN context binary) and profile it.

For each graph: upload -> compile to qnn_dlc (--quantize_full_type float16) -> link to a context binary -> profile on
the device. Job ids go to the state file after every step, so a rerun resumes instead of resubmitting. All graphs
run in parallel threads. The compiled binary's input order is recorded: AI Hub keeps the input names but may
reorder them (and renames the outputs to output_0..N), so the runtime feeds tensors by name.

  python export/aihub_compile.py --onnx-dir work/onnx --state work/aihub_state.json [--device "QCS8550 (Proxy)"]
"""
import argparse
import collections
import fcntl
import json
import os
import sys
import threading
import time

import qai_hub as hub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

LOCK = threading.Lock()


def call(fn, what, timeout=600, tries=20):
    """Retry `fn` with an overall deadline per attempt. The SDK's read timeout is ~3 s and its S3 transfers
    (256 MiB parts, no total timeout) can hang forever behind a proxy, so every call runs in a watchdog thread."""
    for i in range(tries):
        box = {}

        def target():
            try:
                box["r"] = fn()
            except Exception as e:  # noqa: BLE001
                box["e"] = e
        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout)
        if "r" in box:
            return box["r"]
        err = box.get("e", TimeoutError(f"no result after {timeout}s"))
        print(f"  {what}: {type(err).__name__}: {str(err)[:160]} (attempt {i + 1}), retrying", flush=True)
        time.sleep(20)
    raise RuntimeError(f"{what} failed {tries} times")


class State:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            json.dump({}, open(path, "w"))

    def get(self, g):
        return json.load(open(self.path)).get(g, {})

    def set(self, g, **kw):
        with LOCK, open(self.path + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            s = json.load(open(self.path))
            s.setdefault(g, {}).update(kw)
            json.dump(s, open(self.path, "w"), indent=2)


def wait(job_id, label):
    while True:
        st = call(lambda: hub.get_job(job_id).get_status(), f"status {label}")
        if st.finished:
            print(f"[aihub] {label} {job_id} {st.code} {(st.message or '')[:300]}", flush=True)
            if not st.success:
                raise RuntimeError(f"{label} failed: {st.message}")
            return
        time.sleep(60)


def run_graph(g, a, state, device):
    stem = C.GRAPHS[g]
    s = state.get(g)
    if "uploaded" not in s:
        m = call(lambda: hub.upload_model(os.path.join(a.onnx_dir, stem + ".onnx")), f"upload {g}", timeout=7200, tries=3)
        state.set(g, uploaded=m.model_id)
        print(f"[aihub] {g} uploaded {m.model_id}", flush=True)
    s = state.get(g)
    if "link" not in s:
        model = call(lambda: hub.get_model(s["uploaded"]), "get model")
        cj, lj = call(lambda: hub.submit_compile_and_link_jobs(models=model, device=device, name=f"gsv-{stem}-fp16",
                                                               compile_options="--quantize_full_type float16"),
                      f"submit {g}")
        state.set(g, compile=[j.job_id for j in cj], link=lj.job_id)
    s = state.get(g)
    for j in s["compile"]:
        wait(j, f"{g} compile")
    wait(s["link"], f"{g} link")
    if "ctx_model" not in s:
        ctx = call(lambda: hub.get_job(s["link"]).get_target_model(), "get context binary")
        spec = list(ctx.input_spec.values())[0] if isinstance(ctx.input_spec, dict) else []
        state.set(g, ctx_model=ctx.model_id, inputs=[(t.name, list(t.shape), t.dtype) for t in spec])
    s = state.get(g)
    if not a.no_profile and "profile" not in s:
        ctx = call(lambda: hub.get_model(s["ctx_model"]), "get context binary")
        p = call(lambda: hub.submit_profile_job(model=ctx, device=device, name=f"gsv-{stem}-profile"), "submit profile")
        state.set(g, profile=p.job_id)
    s = state.get(g)
    if "profile" in s and "latency_ms" not in s:
        wait(s["profile"], f"{g} profile")
        prof = call(lambda: hub.get_job(s["profile"]).download_profile(), "download profile", timeout=1800)
        es = prof["execution_summary"]
        units = dict(collections.Counter(d.get("compute_unit") for d in prof["execution_detail"]))
        state.set(g, latency_ms=es["estimated_inference_time"] / 1000, peak_mb=es["estimated_inference_peak_memory"] / 2**20,
                  compute_units=units)
    s = state.get(g)
    print(f"[aihub] {g}: {s.get('latency_ms', float('nan')):.2f} ms, {s.get('compute_units')}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", default="work/onnx")
    ap.add_argument("--state", default="work/aihub_state.json")
    ap.add_argument("--device", default="QCS8550 (Proxy)")
    ap.add_argument("--only", nargs="*", default=list(C.GRAPHS), choices=list(C.GRAPHS))
    ap.add_argument("--no-profile", action="store_true")
    a = ap.parse_args()
    state, device = State(a.state), hub.Device(a.device)
    errors = {}

    def worker(g):
        try:
            run_graph(g, a, state, device)
        except Exception as e:  # noqa: BLE001
            errors[g] = e
            print(f"[aihub] {g} FAILED: {e}", flush=True)
    threads = [threading.Thread(target=worker, args=(g,)) for g in a.only]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        sys.exit(f"[aihub] failed: {list(errors)} (rerun to resume; see the AI Hub job logs)")
    print("[aihub] all graphs compiled")


if __name__ == "__main__":
    main()
