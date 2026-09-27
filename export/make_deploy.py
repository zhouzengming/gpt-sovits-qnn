"""Step 6: assemble the device package (copy it to the board and run device/README instructions there).

  <out>/  gsv_runtime.py serve_tts.py text/ src/ Makefile setup_env.sh requirements-device.txt   (from device/)
          manifest.json          graph files, context-binary input order, output names, static shapes
          models/*.bin           QNN context binaries (aihub_download.py)
          assets/                cpu_weights.npz, ref_<voice>.npz, bert_tokenizer/, g2pw_head/
          text/G2PWModel/        G2PW dictionaries (from GPT-SoVITS; the g2pW.onnx itself is not needed)
          qnn_libs/ include/ tools/   QNN runtime from your local QAIRT SDK (not redistributed with this repo)
          lib/libgsv_qnn.so      with --cross-compile (python -m ziglang); otherwise run `make` on the board

  python export/make_deploy.py --work work/myvoice --qairt-sdk qairt/2.50.0.xxxxxx --out dist/myvoice --cross-compile
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
HOST_LIBS = ["libQnnHtp.so", "libQnnSystem.so", "libQnnHtp{arch}Stub.so"]
DSP_LIBS = ["libQnnHtp{arch}Skel.so", "libqnnhtp{arch_l}.cat"]
TOOLS = {"bin": ["qnn-platform-validator"],
         "lib": ["libPlatformValidatorShared.so", "libQnnHtp{arch}CalculatorStub.so", "libcalculator.so"],
         "dsp": ["libCalculator_skel.so"]}
G2PW_DICTS = ["bopomofo_to_pinyin_wo_tune_dict.json", "char_bopomofo_dict.json", "config.py", "MONOPHONIC_CHARS.txt",
              "POLYPHONIC_CHARS.txt", "version"]


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def onnx_outputs(path):
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    return [[o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]] for o in m.graph.output]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="work dir with onnx/, assets/, bin/, aihub_state.json")
    ap.add_argument("--gsv-root", default=os.environ.get("GSV_ROOT"), help="GPT-SoVITS dir (G2PW dictionaries)")
    ap.add_argument("--qairt-sdk", required=True, help="QAIRT SDK root, same version AI Hub compiled with")
    ap.add_argument("--qnn-target", default="aarch64-oe-linux-gcc11.2", help="SDK lib/bin subdir for the board")
    ap.add_argument("--htp-arch", default="v73", help="Hexagon arch of the NPU (QCS8550 / SM8550: v73)")
    ap.add_argument("--model-name", help="model id reported by the TTS service (default: gpt-sovits-<exp>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cross-compile", action="store_true", help="build lib/libgsv_qnn.so with `python -m ziglang`")
    ap.add_argument("--with-onnx", action="store_true", help="also copy the ONNX graphs (for --backend ort tests)")
    a = ap.parse_args()
    if not a.gsv_root:
        raise SystemExit("set --gsv-root or GSV_ROOT (needed for the G2PW dictionaries)")
    work, out, sdk = Path(a.work), Path(a.out), Path(a.qairt_sdk)
    arch, arch_l = a.htp_arch.upper(), a.htp_arch.lower()
    fmt = lambda s: s.format(arch=arch, arch_l=arch_l)  # noqa: E731
    info = json.loads((work / "onnx" / "export.json").read_text())
    state = json.loads((work / "aihub_state.json").read_text())

    shutil.copytree(REPO / "device", out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    g2pw_src = Path(a.gsv_root) / "GPT_SoVITS" / "text" / "G2PWModel"
    for f in G2PW_DICTS:
        copy(g2pw_src / f, out / "text" / "G2PWModel" / f)
    for f in (work / "assets").iterdir():  # cpu_weights.npz, ref_*.npz, bert_tokenizer/
        if f.is_dir():
            shutil.copytree(f, out / "assets" / f.name, dirs_exist_ok=True)
        else:
            copy(f, out / "assets" / f.name)
    shutil.copytree(work / "onnx" / "g2pw_head", out / "assets" / "g2pw_head", dirs_exist_ok=True)
    if not list((out / "assets").glob("ref_*.npz")):
        raise SystemExit(f"no reference voice in {work / 'assets'}: run prepare_assets.py")

    man = {}
    for g, stem in C.GRAPHS.items():
        s = state.get(g, {})
        if "ctx_model" not in s:
            raise SystemExit(f"{g}: not compiled (run aihub_compile.py)")
        copy(work / "bin" / f"{stem}.bin", out / "models" / f"{stem}.bin")
        man[g] = {"file": f"{stem}.bin", "ctx_model": s["ctx_model"], "onnx": f"onnx/{stem}.onnx/model.onnx",
                  "inputs": s["inputs"], "outputs": onnx_outputs(work / "onnx" / f"{stem}.onnx" / "model.onnx"),
                  "latency_ms": s.get("latency_ms")}
        if a.with_onnx:
            shutil.copytree(work / "onnx" / f"{stem}.onnx", out / "onnx" / f"{stem}.onnx", dirs_exist_ok=True)
    exp = (info.get("experiment") or {}).get("name", "voice")
    man["config"] = {"model_name": a.model_name or f"gpt-sovits-{exp}", "shapes": info["shapes"], "t2s": info["t2s"],
                     "sampling_rate": info["sampling_rate"], "hop_length": info["hop_length"],
                     "experiment": info.get("experiment"), "qnn_target": a.qnn_target, "htp_arch": arch_l}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))

    for name in map(fmt, HOST_LIBS):
        copy(sdk / "lib" / a.qnn_target / name, out / "qnn_libs" / name)
    for name in map(fmt, DSP_LIBS):
        src = sdk / "lib" / f"hexagon-{arch_l}" / "unsigned" / name
        if src.exists():
            copy(src, out / "qnn_libs" / name)
    shutil.copytree(sdk / "include" / "QNN", out / "include" / "QNN", dirs_exist_ok=True)
    for name in TOOLS["bin"]:
        copy(sdk / "bin" / a.qnn_target / name, out / "tools" / name)
    for name in map(fmt, TOOLS["lib"]):
        copy(sdk / "lib" / a.qnn_target / name, out / "tools" / "lib" / name)
    for name in TOOLS["dsp"]:
        copy(sdk / "lib" / f"hexagon-{arch_l}" / "unsigned" / name, out / "tools" / "dsp" / name)
    if a.cross_compile:
        (out / "lib").mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-m", "ziglang", "c++", "-target", "aarch64-linux-gnu.2.31", "-std=c++17", "-O2",
                        "-fPIC", "-shared", "-s", f"-I{out / 'include' / 'QNN'}", str(out / "src" / "gsv_qnn.cpp"),
                        "-o", str(out / "lib" / "libgsv_qnn.so"), "-ldl"], check=True)
        print(f"[deploy] cross-compiled {out / 'lib' / 'libgsv_qnn.so'}")
    lines = []
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  ./{f.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"[deploy] package ready: {out} ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 2**30:.2f} GB)")


if __name__ == "__main__":
    main()
