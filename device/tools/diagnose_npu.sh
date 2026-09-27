#!/bin/bash
# Collect everything needed to debug NPU (CDSP / FastRPC) access for QNN HTP on this board.
# Usage: bash tools/diagnose_npu.sh 2>&1 | tee npu_diag.txt
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY="$(dirname "$HERE")"
export LD_LIBRARY_PATH="$HERE/lib:$DEPLOY/qnn_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$HERE/dsp;$DEPLOY/qnn_libs;/usr/lib/rfsa/adsp;/usr/lib/dsp/cdsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp;/dsp"

echo "=== user / groups";            id
echo "=== fastrpc device nodes";     ls -l /dev/adsprpc* /dev/fastrpc* 2>&1
echo "=== fastrpc userspace libs";   ls -l /usr/lib/libcdsprpc* /usr/lib/libadsprpc* /usr/lib/aarch64-linux-gnu/libcdsprpc* 2>&1
echo "=== DSP library dirs";         ls -d /usr/lib/rfsa/adsp /usr/lib/dsp/cdsp /vendor/dsp/cdsp /dsp/cdsp /dsp 2>&1
echo "=== remoteproc state";         for r in /sys/class/remoteproc/remoteproc*; do echo "$r: $(cat $r/name 2>/dev/null) $(cat $r/state 2>/dev/null)"; done
echo "=== kernel log (fastrpc/cdsp)"; (dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null) | grep -i -E "fastrpc|adsprpc|cdsp|remoteproc" | tail -25

echo; echo "=== [1] platform validator as $(whoami)"
"$HERE/qnn-platform-validator" --backend dsp --coreVersion --libVersion --testBackend --targetPath /tmp/qnn_pv_$$ 2>&1 | tail -20

if sudo -n true 2>/dev/null; then
  echo; echo "=== [2] platform validator as root"
  sudo -E env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ADSP_LIBRARY_PATH="$ADSP_LIBRARY_PATH" \
    "$HERE/qnn-platform-validator" --backend dsp --coreVersion --libVersion --testBackend --targetPath /tmp/qnn_pv_root 2>&1 | tail -20
else
  echo; echo "=== [2] skipped (needs passwordless sudo); run manually:"
  echo "sudo -E env LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\" ADSP_LIBRARY_PATH=\"$ADSP_LIBRARY_PATH\" $HERE/qnn-platform-validator --backend dsp --testBackend"
fi
