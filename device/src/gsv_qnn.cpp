// Generic QNN HTP runtime for offline-prepared context binaries, driven from Python through ctypes.
//
// Derived from qnn_reranker.cpp (github.com/zhouzengming/qwen3-reranker-qnn, Apache-2.0), whose loading
// path was validated on a QCS8550 board with QAIRT 2.50:
//   * QCS8550 on Linux needs the aarch64-oe-linux-gcc11.2 QNN libraries ("Unsupported SoC model 66" otherwise)
//   * several single-graph contexts are registered as one HTP context group sharing a spill-fill buffer
//     (QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS); createFromBinaryListAsync + shareResources is
//     rejected by the QCS8550 Linux backend, so it is not used here
//   * burst clock vote as qnn-net-run --perf_profile burst
// Differences: any number of graphs with arbitrary IO; gq_execute() points the QNN tensors straight at the
// caller's buffers (no staging copy on the host), the caller must pass buffers of the exact dtype and size.
// Graph names in AI Hub binaries are random, so graphs are addressed by index (context order, then graph order).

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "HTP/QnnHtpSystemContext.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

namespace {

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList, uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(const QnnSystemInterface_t*** providerList,
                                                                uint32_t* numProviders);

thread_local std::string g_last_error;

void set_error(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_last_error = buf;
}

// ---- Qnn_Tensor_t accessors for the two tensor struct versions ------------------------------
#define GQ_TENSOR_FIELD(t, f) ((t).version == QNN_TENSOR_VERSION_2 ? (t).v2.f : (t).v1.f)

const char* tensor_name(const Qnn_Tensor_t& t) { return GQ_TENSOR_FIELD(t, name); }
Qnn_DataType_t tensor_dtype(const Qnn_Tensor_t& t) { return GQ_TENSOR_FIELD(t, dataType); }
uint32_t tensor_rank(const Qnn_Tensor_t& t) { return GQ_TENSOR_FIELD(t, rank); }
const uint32_t* tensor_dims(const Qnn_Tensor_t& t) { return GQ_TENSOR_FIELD(t, dimensions); }

void tensor_set_name(Qnn_Tensor_t& t, const char* n) {
  if (t.version == QNN_TENSOR_VERSION_2) t.v2.name = n; else t.v1.name = n;
}
void tensor_set_dims(Qnn_Tensor_t& t, uint32_t* d) {
  if (t.version == QNN_TENSOR_VERSION_2) t.v2.dimensions = d; else t.v1.dimensions = d;
}
void tensor_set_raw_buffer(Qnn_Tensor_t& t, void* data, uint32_t size) {
  Qnn_ClientBuffer_t cb{data, size};
  if (t.version == QNN_TENSOR_VERSION_2) {
    t.v2.memType = QNN_TENSORMEMTYPE_RAW;
    t.v2.clientBuf = cb;
    t.v2.isDynamicDimensions = nullptr;  // static graphs only
  } else {
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf = cb;
  }
}

size_t dtype_size(Qnn_DataType_t dt) {
  switch (dt) {
    case QNN_DATATYPE_FLOAT_32: case QNN_DATATYPE_INT_32: case QNN_DATATYPE_UINT_32: return 4;
    case QNN_DATATYPE_FLOAT_16: case QNN_DATATYPE_INT_16: case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: case QNN_DATATYPE_SFIXED_POINT_16: return 2;
    case QNN_DATATYPE_INT_64: case QNN_DATATYPE_UINT_64: return 8;
    case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8: case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_UFIXED_POINT_8: case QNN_DATATYPE_SFIXED_POINT_8: return 1;
    default: return 0;
  }
}

// ---- per-tensor metadata (no buffers: execution uses the caller's memory) -------------------
struct TensorSlot {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  std::string name;
  std::vector<uint32_t> dims;
  uint64_t bytes = 0;

  void init_from(const Qnn_Tensor_t& src) {
    tensor = src;  // shallow copy of the metadata struct, then re-point owned fields
    name = tensor_name(src) ? tensor_name(src) : "";
    dims.assign(tensor_dims(src), tensor_dims(src) + tensor_rank(src));
    uint64_t elements = 1;
    for (uint32_t d : dims) elements *= d;
    bytes = elements * dtype_size(tensor_dtype(src));
    tensor_set_name(tensor, name.c_str());
    tensor_set_dims(tensor, dims.data());
    tensor_set_raw_buffer(tensor, nullptr, 0);
  }
  Qnn_DataType_t dtype() const { return tensor_dtype(tensor); }
};

struct Graph {
  Qnn_GraphHandle_t handle = nullptr;
  std::string name;
  std::vector<TensorSlot> inputs, outputs;
  std::vector<Qnn_Tensor_t> in_array, out_array;
  uint64_t spill_fill_bytes = 0;
};

struct Context {
  std::string path;
  Qnn_ContextHandle_t handle = nullptr;
  std::vector<Graph> graphs;
  uint64_t spill_fill_bytes = 0;
};

struct Runtime {
  void* backend_lib = nullptr;
  void* system_lib = nullptr;
  QNN_INTERFACE_VER_TYPE qnn{};
  QNN_SYSTEM_INTERFACE_VER_TYPE sys{};
  Qnn_LogHandle_t log = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  uint32_t power_config_id = 0;
  bool power_configured = false;
  bool grouped = false;
  std::vector<Context> contexts;
  std::vector<Graph*> graphs;  // flat index: context order, then graph order inside each binary
};

void qnn_log_callback(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  const char* tag = level == QNN_LOG_LEVEL_ERROR ? "ERROR" : level == QNN_LOG_LEVEL_WARN ? "WARN" : "INFO";
  fprintf(stderr, "[QNN %s] ", tag);
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
}

bool load_interfaces(Runtime& rt, const char* backend_path, const char* system_path) {
  rt.backend_lib = dlopen(backend_path, RTLD_NOW | RTLD_GLOBAL);
  if (!rt.backend_lib) { set_error("dlopen(%s) failed: %s", backend_path, dlerror()); return false; }
  auto get_providers = reinterpret_cast<QnnInterfaceGetProvidersFn_t>(dlsym(rt.backend_lib, "QnnInterface_getProviders"));
  if (!get_providers) { set_error("QnnInterface_getProviders not found"); return false; }
  const QnnInterface_t** providers = nullptr;
  uint32_t n = 0;
  if (get_providers(&providers, &n) != QNN_SUCCESS || !providers || n == 0) {
    set_error("no QNN interface providers"); return false;
  }
  bool found = false;
  for (uint32_t i = 0; i < n; i++) {
    if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
        providers[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
      rt.qnn = providers[i]->QNN_INTERFACE_VER_NAME;
      found = true;
      break;
    }
  }
  if (!found) { set_error("backend QNN API version is incompatible with these headers"); return false; }

  rt.system_lib = dlopen(system_path, RTLD_NOW | RTLD_LOCAL);
  if (!rt.system_lib) { set_error("dlopen(%s) failed: %s", system_path, dlerror()); return false; }
  auto get_sys = reinterpret_cast<QnnSystemInterfaceGetProvidersFn_t>(dlsym(rt.system_lib, "QnnSystemInterface_getProviders"));
  if (!get_sys) { set_error("QnnSystemInterface_getProviders not found"); return false; }
  const QnnSystemInterface_t** sys_providers = nullptr;
  if (get_sys(&sys_providers, &n) != QNN_SUCCESS || !sys_providers || n == 0) {
    set_error("no QNN system interface providers"); return false;
  }
  found = false;
  for (uint32_t i = 0; i < n; i++) {
    if (sys_providers[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
        sys_providers[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
      rt.sys = sys_providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
      found = true;
      break;
    }
  }
  if (!found) { set_error("system library API version is incompatible with these headers"); return false; }
  return true;
}

// Vote for maximum HTP clocks ("burst"), as qnn-net-run --perf_profile burst does.
void configure_burst(Runtime& rt) {
  if (!rt.qnn.deviceGetInfrastructure) return;
  QnnDevice_Infrastructure_t infra = nullptr;
  if (rt.qnn.deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) return;
  auto* htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
  if (htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
  QnnHtpDevice_PerfInfrastructure_t perf = htp->perfInfra;
  if (!perf.createPowerConfigId || !perf.setPowerConfig) return;
  if (perf.createPowerConfigId(0, 0, &rt.power_config_id) != QNN_SUCCESS) return;

  QnnHtpPerfInfrastructure_PowerConfig_t dcvs;
  std::memset(&dcvs, 0, sizeof(dcvs));
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  dcvs.dcvsV3Config.contextId = rt.power_config_id;
  dcvs.dcvsV3Config.setDcvsEnable = 1;
  dcvs.dcvsV3Config.dcvsEnable = 0;
  dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  dcvs.dcvsV3Config.setSleepLatency = 1;
  dcvs.dcvsV3Config.sleepLatency = 40;
  dcvs.dcvsV3Config.setSleepDisable = 1;
  dcvs.dcvsV3Config.sleepDisable = 1;
  dcvs.dcvsV3Config.setBusParams = 1;
  dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.setCoreParams = 1;
  dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

  QnnHtpPerfInfrastructure_PowerConfig_t latency;
  std::memset(&latency, 0, sizeof(latency));
  latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  latency.rpcControlLatencyConfig = 100;

  QnnHtpPerfInfrastructure_PowerConfig_t polling;
  std::memset(&polling, 0, sizeof(polling));
  polling.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  polling.rpcPollingTimeConfig = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME;

  const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&dcvs, &latency, &polling, nullptr};
  if (perf.setPowerConfig(rt.power_config_id, configs) == QNN_SUCCESS) {
    rt.power_configured = true;
  } else {
    fprintf(stderr, "[gsv_qnn] warning: could not apply burst power config\n");
  }
}

uint64_t graph_spill_fill(const QnnSystemContext_GraphInfo_t& g) {
  if (g.version != QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3 || !g.graphInfoV3.graphBlobInfo) return 0;
  auto* blob = static_cast<const QnnHtpSystemContext_GraphBlobInfo_t*>(g.graphInfoV3.graphBlobInfo);
  return blob->version == QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1
             ? blob->contextBinaryGraphBlobInfoV1.spillFillBufferSize : 0;
}

uint64_t context_spill_fill(const QnnSystemContext_BinaryInfo_t* info) {
  const void* hw = nullptr;  // V1/V2 binaries: per-context hardware info blob
  if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) hw = info->contextBinaryInfoV1.hwInfoBlob;
  if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) hw = info->contextBinaryInfoV2.hwInfoBlob;
  auto* hwb = static_cast<const QnnHtpSystemContext_HwBlobInfo_t*>(hw);
  return hwb && hwb->version == QNN_SYSTEM_CONTEXT_HTP_HW_INFO_BLOB_VERSION_V1
             ? hwb->contextBinaryHwInfoBlobV1_t.spillFillBufferSize : 0;
}

struct MappedFile {
  void* data = MAP_FAILED;
  uint64_t size = 0;
  explicit MappedFile(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    struct stat st{};
    if (fstat(fd, &st) == 0) {
      size = static_cast<uint64_t>(st.st_size);
      data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    }
    close(fd);
  }
  ~MappedFile() { if (data != MAP_FAILED) munmap(data, size); }
  bool ok() const { return data != MAP_FAILED; }
};

// Graph names + IO tensor metadata from a context binary, without creating a context.
bool read_metadata(Runtime& rt, Context& ctx) {
  MappedFile f(ctx.path);
  if (!f.ok()) { set_error("cannot open/mmap %s", ctx.path.c_str()); return false; }
  QnnSystemContext_Handle_t sys_ctx = nullptr;
  if (rt.sys.systemContextCreate(&sys_ctx) != QNN_SUCCESS) { set_error("systemContextCreate failed"); return false; }
  const QnnSystemContext_BinaryInfo_t* info = nullptr;
  Qnn_ContextBinarySize_t info_size = 0;
  bool ok = rt.sys.systemContextGetBinaryInfo(sys_ctx, f.data, f.size, &info, &info_size) == QNN_SUCCESS && info;
  if (!ok) set_error("systemContextGetBinaryInfo failed for %s", ctx.path.c_str());
  uint32_t num_graphs = 0;
  const QnnSystemContext_GraphInfo_t* graphs = nullptr;
  if (ok) {
    switch (info->version) {
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: num_graphs = info->contextBinaryInfoV1.numGraphs; graphs = info->contextBinaryInfoV1.graphs; break;
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: num_graphs = info->contextBinaryInfoV2.numGraphs; graphs = info->contextBinaryInfoV2.graphs; break;
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: num_graphs = info->contextBinaryInfoV3.numGraphs; graphs = info->contextBinaryInfoV3.graphs; break;
      default: ok = false; set_error("unsupported binary info version %d", static_cast<int>(info->version));
    }
  }
  if (ok && (num_graphs == 0 || !graphs)) { ok = false; set_error("%s: no graphs in binary", ctx.path.c_str()); }
  if (ok) {
    ctx.graphs.resize(num_graphs);  // sized once: tensors keep pointers into TensorSlot storage
    uint64_t ctx_spill = context_spill_fill(info);
    for (uint32_t gi = 0; gi < num_graphs && ok; gi++) {
      const auto& g = graphs[gi];
      const char* name = nullptr;
      const Qnn_Tensor_t *in = nullptr, *out = nullptr;
      uint32_t n_in = 0, n_out = 0;
      if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        name = g.graphInfoV1.graphName; in = g.graphInfoV1.graphInputs; n_in = g.graphInfoV1.numGraphInputs;
        out = g.graphInfoV1.graphOutputs; n_out = g.graphInfoV1.numGraphOutputs;
      } else if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        name = g.graphInfoV2.graphName; in = g.graphInfoV2.graphInputs; n_in = g.graphInfoV2.numGraphInputs;
        out = g.graphInfoV2.graphOutputs; n_out = g.graphInfoV2.numGraphOutputs;
      } else if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        name = g.graphInfoV3.graphName; in = g.graphInfoV3.graphInputs; n_in = g.graphInfoV3.numGraphInputs;
        out = g.graphInfoV3.graphOutputs; n_out = g.graphInfoV3.numGraphOutputs;
      } else {
        ok = false; set_error("unsupported graph info version %d", static_cast<int>(g.version)); break;
      }
      Graph& gr = ctx.graphs[gi];
      gr.name = name ? name : "";
      gr.spill_fill_bytes = graph_spill_fill(g);
      if (gr.spill_fill_bytes == 0) gr.spill_fill_bytes = ctx_spill;
      ctx.spill_fill_bytes = std::max(ctx.spill_fill_bytes, gr.spill_fill_bytes);
      gr.inputs.resize(n_in);
      gr.outputs.resize(n_out);
      for (uint32_t i = 0; i < n_in; i++) gr.inputs[i].init_from(in[i]);
      for (uint32_t i = 0; i < n_out; i++) gr.outputs[i].init_from(out[i]);
    }
  }
  rt.sys.systemContextFree(sys_ctx);
  return ok;
}

Qnn_ErrorHandle_t create_context(Runtime& rt, Context& ctx, const MappedFile& f, Qnn_ContextHandle_t group_first,
                                 uint64_t spill) {
  QnnHtpContext_CustomConfig_t group_cfg;
  std::memset(&group_cfg, 0, sizeof(group_cfg));
  group_cfg.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
  group_cfg.groupRegistration.firstGroupHandle = group_first;
  group_cfg.groupRegistration.maxSpillFillBuffer = spill;
  QnnContext_Config_t cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
  cfg.customConfig = &group_cfg;
  const QnnContext_Config_t* cfgs[] = {&cfg, nullptr};
  return rt.qnn.contextCreateFromBinary(rt.backend, rt.device, spill ? cfgs : nullptr, f.data, f.size, &ctx.handle, nullptr);
}

// With several contexts, try to join a shared spill-fill group first; fall back to a standalone context
// if the backend rejects the group option (e.g. the x86 simulator).
bool load_context(Runtime& rt, Context& ctx, Qnn_ContextHandle_t group_first, uint64_t spill, int log_level) {
  MappedFile f(ctx.path);
  if (!f.ok()) { set_error("cannot open/mmap %s", ctx.path.c_str()); return false; }
  Qnn_ErrorHandle_t err = QNN_SUCCESS;
  if (spill && (rt.grouped || !group_first)) {
    err = create_context(rt, ctx, f, group_first, spill);
    if (err == QNN_SUCCESS) {
      rt.grouped = true;
    } else if (!group_first) {
      if (log_level >= 1)
        fprintf(stderr, "[gsv_qnn] warning: shared spill-fill group not supported (err %lu); loading contexts standalone\n",
                static_cast<unsigned long>(err));
      ctx.handle = nullptr;
      err = create_context(rt, ctx, f, nullptr, 0);
    }
  } else {
    err = create_context(rt, ctx, f, nullptr, 0);
  }
  if (err != QNN_SUCCESS) {
    set_error("contextCreateFromBinary(%s) failed: %lu%s", ctx.path.c_str(), static_cast<unsigned long>(err),
              (group_first && !rt.grouped) ? " (contexts not sharing memory: the HTP process domain is probably exhausted)" : "");
    return false;
  }
  for (auto& g : ctx.graphs) {
    if (rt.qnn.graphRetrieve(ctx.handle, g.name.c_str(), &g.handle) != QNN_SUCCESS) {
      set_error("graphRetrieve(%s) failed", g.name.c_str());
      return false;
    }
    for (auto& t : g.inputs) g.in_array.push_back(t.tensor);
    for (auto& t : g.outputs) g.out_array.push_back(t.tensor);
  }
  return true;
}

void destroy(Runtime* rt) {
  if (!rt) return;
  // grouped contexts: release the members before the first context that owns the shared buffer
  for (auto it = rt->contexts.rbegin(); it != rt->contexts.rend(); ++it) {
    if (it->handle && rt->qnn.contextFree) rt->qnn.contextFree(it->handle, nullptr);
  }
  if (rt->power_configured) {
    QnnDevice_Infrastructure_t infra = nullptr;
    if (rt->qnn.deviceGetInfrastructure && rt->qnn.deviceGetInfrastructure(&infra) == QNN_SUCCESS && infra) {
      auto* htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
      if (htp->perfInfra.destroyPowerConfigId) htp->perfInfra.destroyPowerConfigId(rt->power_config_id);
    }
  }
  if (rt->device && rt->qnn.deviceFree) rt->qnn.deviceFree(rt->device);
  if (rt->backend && rt->qnn.backendFree) rt->qnn.backendFree(rt->backend);
  if (rt->log && rt->qnn.logFree) rt->qnn.logFree(rt->log);
  if (rt->system_lib) dlclose(rt->system_lib);
  // the backend library is intentionally not dlclose'd: HTP keeps worker threads around
  delete rt;
}

Graph* graph_at(void* h, int g) {
  auto* rt = static_cast<Runtime*>(h);
  if (!rt || g < 0 || g >= static_cast<int>(rt->graphs.size())) { set_error("graph index %d out of range", g); return nullptr; }
  return rt->graphs[g];
}

}  // namespace

extern "C" {

const char* gq_last_error() { return g_last_error.c_str(); }

// ctx_paths : context binaries; graphs are numbered in this order (then by their order inside each binary)
// perf_mode : 1 = burst clocks (recommended), 0 = leave DCVS defaults
// log_level : 0 = off, 1 = error, 2 = warn, 3 = info
void* gq_create(const char* backend_path, const char* system_path, const char** ctx_paths, int n_ctx, int perf_mode,
                int log_level) {
  auto* rt = new Runtime();
  if (!load_interfaces(*rt, backend_path, system_path)) { destroy(rt); return nullptr; }
  if (log_level > 0 && rt->qnn.logCreate) {
    auto lvl = log_level >= 3 ? QNN_LOG_LEVEL_INFO : log_level == 2 ? QNN_LOG_LEVEL_WARN : QNN_LOG_LEVEL_ERROR;
    rt->qnn.logCreate(qnn_log_callback, lvl, &rt->log);
  }
  if (rt->qnn.backendCreate(rt->log, nullptr, &rt->backend) != QNN_SUCCESS) {
    set_error("backendCreate failed"); destroy(rt); return nullptr;
  }
  if (rt->qnn.deviceCreate && rt->qnn.deviceCreate(rt->log, nullptr, &rt->device) != QNN_SUCCESS) {
    set_error("deviceCreate failed (check the QNN log above; 'Unsupported SoC model' means the runtime libraries do "
              "not support this SoC - QCS8550 needs the aarch64-oe-linux-gcc11.2 build)");
    destroy(rt); return nullptr;
  }
  if (perf_mode == 1) configure_burst(*rt);

  rt->contexts.resize(n_ctx);  // sized once: graphs are referenced by pointer below
  uint64_t spill = 0;
  for (int i = 0; i < n_ctx; i++) {
    rt->contexts[i].path = ctx_paths[i];
    if (!read_metadata(*rt, rt->contexts[i])) { destroy(rt); return nullptr; }
    spill = std::max(spill, rt->contexts[i].spill_fill_bytes);
  }
  if (log_level >= 2) fprintf(stderr, "[gsv_qnn] %d context(s), max spill-fill %.1f MiB\n", n_ctx, spill / 1048576.0);
  for (int i = 0; i < n_ctx; i++) {
    Qnn_ContextHandle_t first = i == 0 ? nullptr : rt->contexts[0].handle;
    if (!load_context(*rt, rt->contexts[i], first, n_ctx > 1 ? spill : 0, log_level)) { destroy(rt); return nullptr; }
  }
  for (auto& c : rt->contexts)
    for (auto& g : c.graphs) rt->graphs.push_back(&g);
  return rt;
}

void gq_destroy(void* h) { destroy(static_cast<Runtime*>(h)); }

int gq_num_graphs(void* h) { return static_cast<int>(static_cast<Runtime*>(h)->graphs.size()); }

// 1 = contexts registered as one spill-fill group, 0 = standalone contexts
int gq_grouped(void* h) { return static_cast<Runtime*>(h)->grouped ? 1 : 0; }

const char* gq_graph_name(void* h, int g) {
  Graph* gr = graph_at(h, g);
  return gr ? gr->name.c_str() : "";
}

unsigned long long gq_spill_fill_bytes(void* h, int g) {
  Graph* gr = graph_at(h, g);
  return gr ? gr->spill_fill_bytes : 0;
}

int gq_num_tensors(void* h, int g, int is_output) {
  Graph* gr = graph_at(h, g);
  if (!gr) return -1;
  return static_cast<int>(is_output ? gr->outputs.size() : gr->inputs.size());
}

// Describe one IO tensor. dtype is the raw Qnn_DataType_t value. Returns 0 on success.
int gq_tensor_info(void* h, int g, int is_output, int index, char* name, int name_len, int* dtype, uint32_t* dims,
                   int* rank, unsigned long long* bytes) {
  Graph* gr = graph_at(h, g);
  if (!gr) return -1;
  auto& v = is_output ? gr->outputs : gr->inputs;
  if (index < 0 || index >= static_cast<int>(v.size())) { set_error("tensor index %d out of range", index); return -1; }
  const auto& t = v[index];
  if (name && name_len > 0) snprintf(name, name_len, "%s", t.name.c_str());
  if (dtype) *dtype = static_cast<int>(t.dtype());
  if (rank) *rank = static_cast<int>(t.dims.size());
  if (dims) for (size_t i = 0; i < t.dims.size() && i < 8; i++) dims[i] = t.dims[i];
  if (bytes) *bytes = t.bytes;
  return 0;
}

// Execute graph g. inputs/outputs: one caller buffer per tensor, in the binary's tensor order, each exactly
// gq_tensor_info().bytes long and of the tensor's dtype. QNN reads/writes these buffers directly.
int gq_execute(void* h, int g, void* const* inputs, void* const* outputs, double* ms) {
  auto* rt = static_cast<Runtime*>(h);
  Graph* gr = graph_at(h, g);
  if (!gr) return -1;
  for (size_t i = 0; i < gr->inputs.size(); i++) {
    if (!inputs[i]) { set_error("graph %d: input %zu is null", g, i); return -1; }
    tensor_set_raw_buffer(gr->in_array[i], inputs[i], static_cast<uint32_t>(gr->inputs[i].bytes));
  }
  for (size_t i = 0; i < gr->outputs.size(); i++) {
    if (!outputs[i]) { set_error("graph %d: output %zu is null", g, i); return -1; }
    tensor_set_raw_buffer(gr->out_array[i], outputs[i], static_cast<uint32_t>(gr->outputs[i].bytes));
  }
  auto t0 = std::chrono::steady_clock::now();
  Qnn_ErrorHandle_t err = rt->qnn.graphExecute(gr->handle, gr->in_array.data(), static_cast<uint32_t>(gr->in_array.size()),
                                               gr->out_array.data(), static_cast<uint32_t>(gr->out_array.size()), nullptr, nullptr);
  auto t1 = std::chrono::steady_clock::now();
  if (err != QNN_GRAPH_NO_ERROR) { set_error("graphExecute failed on graph %d (%s): %lu", g, gr->name.c_str(), static_cast<unsigned long>(err)); return -1; }
  if (ms) *ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return 0;
}

}  // extern "C"
