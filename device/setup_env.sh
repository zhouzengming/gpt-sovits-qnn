# Source this on the board before running:  source setup_env.sh
# - LD_LIBRARY_PATH : QNN host libraries (libQnnHtp.so, libQnnSystem.so, libQnnHtpV73Stub.so)
# - ADSP_LIBRARY_PATH : where FastRPC looks for the DSP-side skeleton (libQnnHtpV73Skel.so);
#   the system DSP library dirs are kept so other DSP users keep working.
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$DEPLOY_DIR/qnn_libs:$DEPLOY_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$DEPLOY_DIR/qnn_libs;/usr/lib/rfsa/adsp;/usr/lib/dsp/cdsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp;/dsp"
