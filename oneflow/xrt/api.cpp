#include "oneflow/xrt/api.h"
#include "glog/logging.h"
#include "oneflow/core/operator/operator.h"  // GenLogicalBlobName, GenLogicalBlobId
#include "oneflow/xrt/build_graph.h"
#include "oneflow/xrt/utility/env.h"

DEFINE_int32(clustering_minimum_nodes,
             EnvToInt(FLAGS_clustering_minimum_nodes, 1),
             "Minium nodes of a cluster after clustering.");
DEFINE_int32(clustering_maximum_nodes,
             EnvToInt(FLAGS_clustering_maximum_nodes, 1000),
             "Maxium nodes of a cluster after clustering.");
DEFINE_bool(strict_clustering, EnvToBool(FLAGS_strict_clustering, true),
            "Option to clustering with strict dependencies analysis.");

namespace oneflow {
namespace xrt {

#define OP_TYPE_CASE(op) OperatorConf::k##op##Conf

static std::unordered_map<int32_t, std::string> op_type2string_map = {
    {OP_TYPE_CASE(Matmul), "MatMul"},
    {OP_TYPE_CASE(Relu), "Relu"},
    // {OP_TYPE_CASE(FullyConnected), "FullyConnected"},
    {OP_TYPE_CASE(BiasAdd), "BiasAdd"},
    {OP_TYPE_CASE(Reshape), "Reshape"},
    {OP_TYPE_CASE(Identity), "Identity"},
    {OP_TYPE_CASE(ReshapeLike), "ReshapeLike"},
    {OP_TYPE_CASE(Cast), "Cast"},
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
    {OP_TYPE_CASE(AdamModelUpdate), "AdamModelUpdate"},
    {OP_TYPE_CASE(AdamOptimizer), "AdamOptimizer"},
    {OP_TYPE_CASE(ClipGradient), "ClipGradient"},
    {OP_TYPE_CASE(ReduceConcat), "ReduceConcat"},
    {OP_TYPE_CASE(ReduceSplit), "ReduceSplit"},
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
    case DeviceType::kGPU:
      return XrtDevice::GPU_CUDA;
    case DeviceType::kCPU:
      return XrtDevice::CPU_X86;
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
    LOG(FATAL) << "Can not convert xrt device (" << device
               << ") to device type.";
    return DeviceType::kCPU;
  }
}

std::string BlobIdToName(const LogicalBlobId &lbi) {
  CHECK_EQ(lbi.has_op_name(), true);
  CHECK_EQ(lbi.has_blob_name(), true);
  if (lbi.op_name() == "") {
    return lbi.blob_name();
  }
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

std::shared_ptr<XrtGraph> BuildXrtGraph(const XrtLaunchOpConf &launch_conf,
                                        const DeviceType &device_type,
                                        const JobDesc &job_desc) {
  return graph_builder::BuildGraph(launch_conf, device_type, job_desc);
}

XrtPassOptions CreateDefaultXrtPassOptions() {
  ClusteringOptions options;
  options.minimum_nodes = FLAGS_clustering_minimum_nodes;
  options.maximum_nodes = FLAGS_clustering_maximum_nodes;
  options.strict_clustering = FLAGS_strict_clustering;

  XrtPassOptions xrt_options;
  xrt_options.clustering_options = options;
  return xrt_options;
}

Parameter BuildParameter(const Blob &blob, const std::string &name) {
  const auto &desc = blob.blob_desc();
  return Parameter(name, const_cast<void *>(blob.dptr<void>()), desc.shape(),
                   desc.data_type());
}

bool LookupMutability(const XrtLaunchOpConf &launch_conf,
                      const std::string &argument) {
  const auto &mutability_table = launch_conf.attr().mutability();
  const auto &it = mutability_table.find(argument);
  if (it == mutability_table.end()) {
    return false;
  }
  return it->second;
}

}  // namespace xrt
}  // namespace oneflow
