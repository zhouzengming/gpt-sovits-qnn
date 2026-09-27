#!/usr/bin/env bash
# 完整转换流程：GPT-SoVITS 训练目录 logs/<实验名> -> 静态 ONNX -> AI Hub fp16 QNN context binary -> 板端部署包
#
#   1. 把 GPT-SoVITS 的 logs/<实验名>/ 整个复制到本仓库的 logs/ 下（见 logs/README.md）
#   2. 把 QAIRT SDK 解压到本仓库的 qairt/ 下（见 qairt/README.md）
#   3. GSV_ROOT=/path/to/GPT-SoVITS bash scripts/run_pipeline.sh <实验名>
#
# GSV_ROOT：训练用的 GPT-SoVITS 目录（需要其中的代码和 GPT_SoVITS/pretrained_models 下的预训练模型）。
# 可通过环境变量调整（方括号内为默认值）：
#   GPT_EPOCH / SOVITS_EPOCH [最后一轮]   REF [自动选一条 3~10 秒的训练音频，也可指定 5-wav32k 下的文件名]
#   VOICE [实验名]   DEVICE ["QCS8550 (Proxy)"]   QNN_TARGET [aarch64-oe-linux-gcc11.2]   HTP_ARCH [v73]
#   WORK [work/<实验名>]   OUT [dist/<实验名>]   QAIRT_SDK [在 qairt/ 下自动查找]
#   EVAL [0]（设为 1 时额外在 AI Hub 真机上验证编译结果，部署包会同时带上 ONNX）
set -euo pipefail
cd "$(dirname "$0")/.."

EXP="${1:-}"
if [ -z "$EXP" ] || [ ! -d "logs/$EXP" ]; then
  echo "用法：GSV_ROOT=/path/to/GPT-SoVITS bash scripts/run_pipeline.sh <实验名>" >&2
  echo "      其中 logs/<实验名>/ 是从 GPT-SoVITS 的 logs/ 下复制过来的训练目录。现有：$(ls logs 2>/dev/null | grep -v README | tr '\n' ' ')" >&2
  exit 1
fi
[ -n "${GSV_ROOT:-}" ] && [ -d "$GSV_ROOT/GPT_SoVITS/pretrained_models" ] || {
  echo "请设置 GSV_ROOT 为训练用的 GPT-SoVITS 目录（需包含 GPT_SoVITS/pretrained_models）。" >&2; exit 1; }
export GSV_ROOT

# QAIRT SDK：优先使用环境变量 QAIRT_SDK，否则在仓库的 qairt/ 目录下自动查找（见 qairt/README.md）
if [ -z "${QAIRT_SDK:-}" ]; then
  mapfile -t _sdks < <(find -L qairt -maxdepth 6 -path "*/include/QNN/QnnInterface.h" 2>/dev/null \
                         | sed 's#/include/QNN/QnnInterface.h$##' | sort)
  if [ "${#_sdks[@]}" -eq 0 ]; then
    echo "未找到 QAIRT SDK：请按 qairt/README.md 把 SDK 解压到 qairt/ 目录，或设置环境变量 QAIRT_SDK。" >&2
    exit 1
  elif [ "${#_sdks[@]}" -gt 1 ]; then
    echo "qairt/ 下找到多个 SDK，请用 QAIRT_SDK 指定其中一个：" >&2
    printf '  %s\n' "${_sdks[@]}" >&2
    exit 1
  fi
  QAIRT_SDK="$(cd "${_sdks[0]}" && pwd)"
fi
[ -f "$QAIRT_SDK/include/QNN/QnnInterface.h" ] || { echo "QAIRT_SDK=$QAIRT_SDK 不是有效的 SDK 根目录" >&2; exit 1; }
_ver="$(grep -E '^version:' "$QAIRT_SDK/sdk.yaml" 2>/dev/null | awk '{print $2}')"
echo "== QAIRT SDK：$QAIRT_SDK（版本 ${_ver:-未知}，需与 AI Hub 编译时的版本一致）"

WORK="${WORK:-work/$EXP}"
OUT="${OUT:-dist/$EXP}"
DEVICE="${DEVICE:-QCS8550 (Proxy)}"
QNN_TARGET="${QNN_TARGET:-aarch64-oe-linux-gcc11.2}"
HTP_ARCH="${HTP_ARCH:-v73}"
PY="${PYTHON:-python3}"
SEL=(--exp "logs/$EXP")
[ -n "${GPT_EPOCH:-}" ] && SEL+=(--gpt-epoch "$GPT_EPOCH")
[ -n "${SOVITS_EPOCH:-}" ] && SEL+=(--sovits-epoch "$SOVITS_EPOCH")
mkdir -p "$WORK"

echo "== [1/6] 导出静态 ONNX（6 个图）"
[ -f "$WORK/onnx/export.json" ] || $PY export/export_onnx.py "${SEL[@]}" --out "$WORK/onnx"

echo "== [2/6] 与 GPT-SoVITS 原实现对比验证（CPU）"
$PY export/verify_onnx.py "${SEL[@]}" --onnx-dir "$WORK/onnx"

echo "== [3/6] 生成 CPU 侧权重与参考音色"
REF_ARGS=()
[ -n "${REF:-}" ] && REF_ARGS+=(--ref "$REF")
[ -n "${VOICE:-}" ] && REF_ARGS+=(--voice "$VOICE")
$PY export/prepare_assets.py "${SEL[@]}" "${REF_ARGS[@]}" --out "$WORK/assets"

echo "== [4/6] 在 AI Hub 上为 $DEVICE 编译 fp16 QNN context binary 并测速"
$PY export/aihub_compile.py --onnx-dir "$WORK/onnx" --state "$WORK/aihub_state.json" --device "$DEVICE"

echo "== [5/6] 下载 context binary"
$PY export/aihub_download.py --state "$WORK/aihub_state.json" --out "$WORK/bin"

echo "== [6/6] 组装板端部署包 -> $OUT"
EXTRA=()
$PY -c "import ziglang" 2>/dev/null && EXTRA+=(--cross-compile)
[ "${EVAL:-0}" = "1" ] && EXTRA+=(--with-onnx)
$PY export/make_deploy.py --work "$WORK" --qairt-sdk "$QAIRT_SDK" --qnn-target "$QNN_TARGET" --htp-arch "$HTP_ARCH" \
    --out "$OUT" "${EXTRA[@]}"

if [ "${EVAL:-0}" = "1" ]; then
  echo "== [可选] 在 AI Hub 真机上验证 context binary"
  $PY export/aihub_verify.py --deploy "$OUT" --device "$DEVICE"
fi

echo "完成。把 $OUT 拷到开发板上，然后执行（详见 README）："
echo "  pip install -r requirements-device.txt && source setup_env.sh"
echo "  python gsv_runtime.py --backend qnn --text \"今天天气不错。\" --out out.wav"
echo "  python serve_tts.py --host 0.0.0.0 --port 8000"
