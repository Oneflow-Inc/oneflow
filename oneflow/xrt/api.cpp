#include "oneflow/xrt/api.h"

#include "glog/logging.h"

#include "oneflow/core/operator/operator.h"  // GenLogicalBlobName, GenLogicalBlobId
#include "oneflow/xrt/build_graph.h"
#include "oneflow/xrt/utility/env.h"

DEFINE_int32(clustering_minimum_nodes, EnvToInt(FLAGS_clustering_minimum_nodes, 1),
             "Minium nodes of a cluster after clustering.");
DEFINE_int32(clustering_maximum_nodes, EnvToInt(FLAGS_clustering_maximum_nodes, 1000),
             "Maxium nodes of a cluster after clustering.");
DEFINE_bool(strict_clustering, EnvToBool(FLAGS_strict_clustering, true),
            "Option to clustering with strict dependencies analysis.");

// DEFINE_string(engine, EnvToString(FLAGS_engine, "XLA"),
//               "Which third party engine to be used. XLA and TENSORRT are "
//               "valid, Default means using no engine.");
DEFINE_bool(use_xla_jit, EnvToBool(FLAGS_use_xla_jit, false), "It's optional to use xla jit.");
DEFINE_bool(use_tensorrt, EnvToBool(FLAGS_use_tensorrt, false), "It's optional to use tensorrt.");

DEFINE_bool(tensorrt_fp16, EnvToBool(FLAGS_tensorrt_fp16, false),
            "Enable fp16 precision for TENSORRT engine.");
DEFINE_bool(tensorrt_int8, EnvToBool(FLAGS_tensorrt_int8, false),
            "Enable int8 precision for TENSORRT engine.");

namespace oneflow {
namespace xrt {

#define OP_TYPE_CASE(op) OperatorConf::k##op##Conf

static std::unordered_map<int32_t, std::string> op_type2string_map = {
    {OP_TYPE_CASE(Matmul), "MatMul"},
    {OP_TYPE_CASE(Relu), "Relu"},
    {OP_TYPE_CASE(LeakyRelu), "LeakyRelu"},
    {OP_TYPE_CASE(Conv2D), "Conv2D"},
    {OP_TYPE_CASE(Multiply), "Multiply"},
    // {OP_TYPE_CASE(FullyConnected), "FullyConnected"},
    {OP_TYPE_CASE(BiasAdd), "BiasAdd"},
    {OP_TYPE_CASE(Reshape), "Reshape"},
    {OP_TYPE_CASE(Identity), "Identity"},
    {OP_TYPE_CASE(ReshapeLike), "ReshapeLike"},
    {OP_TYPE_CASE(Cast), "Cast"},
    {OP_TYPE_CASE(Concat), "Concat"},
    {OP_TYPE_CASE(ScalarAdd), "ScalarAdd"},
    {OP_TYPE_CASE(ScalarMul), "ScalarMul"},
    {OP_TYPE_CASE(Transpose), "Transpose"},
    {OP_TYPE_CASE(BroadcastAdd), "BcastAdd"},
    {OP_TYPE_CASE(BroadcastMul), "BcastMul"},
    {OP_TYPE_CASE(BroadcastDiv), "BcastDiv"},
    {OP_TYPE_CASE(Add), "Add"},
    {OP_TYPE_CASE(Sigmoid), "Sigmoid"},
    {OP_TYPE_CASE(Tanh), "Tanh"},
    {OP_TYPE_CASE(TanhGrad), "TanhGrad"},
    {OP_TYPE_CASE(Gelu), "Gelu"},
    {OP_TYPE_CASE(GeluGrad), "GeluGrad"},
    {OP_TYPE_CASE(Gather), "Gather"},
    {OP_TYPE_CASE(BatchGather), "BatchGather"},
    {OP_TYPE_CASE(Softmax), "Softmax"},
    {OP_TYPE_CASE(SoftmaxGrad), "SoftmaxGrad"},
    {OP_TYPE_CASE(LayerNorm), "LayerNorm"},
    {OP_TYPE_CASE(LayerNormParamGrad), "LayerNormParamGrad"},
    {OP_TYPE_CASE(LayerNormGrad), "LayerNormGrad"},
    {OP_TYPE_CASE(ReduceSum), "ReduceSum"},
    {OP_TYPE_CASE(ReduceMean), "ReduceMean"},
    {OP_TYPE_CASE(AdamModelUpdate), "AdamOptimizer"},
    {OP_TYPE_CASE(MaxPooling2D), "MaxPooling2D"},
    {OP_TYPE_CASE(AveragePooling2D), "AveragePooling2D"},
    {OP_TYPE_CASE(Normalization), "Normalization"},
    // {OP_TYPE_CASE(ReduceConcat), "ReduceConcat"},
    // {OP_TYPE_CASE(ReduceSplit), "ReduceSplit"},
    // TODO(hjchen2)
};

std::string ExtractOpTypeAsString(const OperatorConf &conf) {
  const auto it = op_type2string_map.find(conf.op_type_case());
  if (it != op_type2string_map.end()) {
    return it->second;
  } else {
    // Return empty if the operator is not in the translation map
    return std::string("");
  }
}

XrtDevice DeviceTypeToXrtDevice(const DeviceType &device_type) {
  switch (device_type) {
    case DeviceType::kGPU: return XrtDevice::GPU_CUDA;
    case DeviceType::kCPU: return XrtDevice::CPU_X86;
    default:
      DLOG(WARNING) << "Meet invalid device type (" << device_type
                    << "). Use the default xrt device instead.";
      return XrtDevice::CPU_X86;
  }
}

DeviceType XrtDeviceToDeviceType(const XrtDevice &device) {
  if (device == XrtDevice::GPU_CUDA) {
    return DeviceType::kGPU;
  } else if (device == XrtDevice::CPU_X86) {
    return DeviceType::kCPU;
  } else {
    LOG(FATAL) << "Can not convert xrt device (" << device << ") to device type.";
    return DeviceType::kCPU;
  }
}

XrtEngine StringToXrtEngine(const std::string &engine) {
  if (engine == "XLA") {
    return xrt::XrtEngine::XLA;
  } else if (engine == "TENSORRT") {
    return xrt::XrtEngine::TENSORRT;
  } else {
    LOG(FATAL) << "Unknown engine: " << engine;
  }
}

std::string BlobIdToName(const LogicalBlobId &lbi) {
  CHECK_EQ(lbi.has_op_name(), true);
  CHECK_EQ(lbi.has_blob_name(), true);
  if (lbi.op_name() == "") { return lbi.blob_name(); }
  return GenLogicalBlobName(lbi);
}

LogicalBlobId BlobNameToId(const std::string &blob_name) {
  size_t pos = blob_name.find('/');
  if (pos == std::string::npos) {
    return GenLogicalBlobId("/" + blob_name);
  } else {
    return GenLogicalBlobId(blob_name);
  }
}

std::shared_ptr<XrtGraph> BuildXrtGraph(const OpGraph *op_graph) {
  return graph_builder::BuildGraph(op_graph);
}

std::shared_ptr<XrtGraph> BuildXrtGraph(const XrtLaunchOpConf::Function &function,
                                        const DeviceType &device_type, const JobDesc &job_desc) {
  return graph_builder::BuildGraph(function, device_type, job_desc);
}

void InitXrtConfigurations(const XrtConfig &config) {
  if (config.has_use_xla_jit()) { FLAGS_use_xla_jit = config.use_xla_jit(); }
  if (config.has_use_tensorrt()) { FLAGS_use_tensorrt = config.use_tensorrt(); }
  // Set xla configurations.
  if (config.has_tensorrt_config()) {
    const XrtConfig::TensorRTConfig &trt_config = config.tensorrt_config();
    if (trt_config.has_use_fp16()) { FLAGS_tensorrt_fp16 = trt_config.use_fp16(); }
    if (trt_config.has_use_int8()) { FLAGS_tensorrt_int8 = trt_config.use_int8(); }
  }
}

bool XrtCompilationEnabled() { return FLAGS_use_xla_jit || FLAGS_use_tensorrt; }

XrtPassOptions CreateDefaultXrtPassOptions(bool train_phase) {
  ClusteringOptions options;
  options.minimum_nodes = FLAGS_clustering_minimum_nodes;
  options.maximum_nodes = FLAGS_clustering_maximum_nodes;
  options.strict_clustering = FLAGS_strict_clustering;

  options.train_phase = train_phase;
  // TODO(hjchen2)
  options.engine = (1U << XrtEngineOptionBit::kUseDefault);
  if (FLAGS_use_xla_jit) { options.engine |= (1U << XrtEngineOptionBit::kUseXlaJit); }
  if (FLAGS_use_tensorrt) { options.engine |= (1U << XrtEngineOptionBit::kUseTensorRT); }

  XrtPassOptions xrt_options;
  xrt_options.clustering_options = options;
  return xrt_options;
}

void RunCompilationTimeXrtPasses(const OpGraph &op_graph, Job *job, bool train_phase) {
  auto graph = BuildXrtGraph(&op_graph);
  // Create options to run xrt passes.
  auto options = CreateDefaultXrtPassOptions(train_phase);

  RunXrtPass("MarkClusterId", graph.get(), options);
  RunXrtPass("BuildSubGraph", graph.get(), options);
  // Rebuild Job
  RunXrtPass("RebuildCompiledJob", graph.get(), options, job);
}

}  // namespace xrt
}  // namespace oneflow
