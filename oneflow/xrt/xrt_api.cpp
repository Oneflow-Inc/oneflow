#include "oneflow/xrt/xrt_api.h"
#include "glog/logging.h"
#include "oneflow/core/operator/operator.h"  // GenLogicalBlobName, GenLogicalBlobId
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

extern std::string _XrtArgumentOpType;

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

XrtDevice DeviceTypeToBackend(const DeviceType &device_type) {
  switch (device_type) {
    case DeviceType::kGPU:
      return XrtDevice::GPU_CUDA;
    case DeviceType::kCPU:
      return XrtDevice::CPU_X86;
    default:
      DLOG(WARNING) << "Meet invalid device type (" << device_type
                    << "). Use the default backend instead.";
      return XrtDevice::CPU_X86;
  }
}

DeviceType BackendToDeviceType(const XrtDevice &backend) {
  if (backend == XrtDevice::GPU_CUDA) {
    return DeviceType::kGPU;
  } else if (backend == XrtDevice::CPU_X86) {
    return DeviceType::kCPU;
  } else {
    LOG(FATAL) << "Can not convert backend (" << backend << ") to device type.";
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

const Shape &InputTimeShape(const OpNode *op_node) {
  CHECK_NOTNULL(op_node);
  return *(op_node->GetInputBlobFastestTimeShape());
}

const Shape &OutputTimeShape(const OpNode *op_node) {
  CHECK_NOTNULL(op_node);
  return *(op_node->out_blob_time_shape());
}

const SbpParallel &BlobSbpPolicy(const OpNode *op_node,
                                 const std::string &name) {
  CHECK_NOTNULL(op_node);
  LogicalBlobId lbi = BlobNameToId(name);
  return op_node->SbpParallel4Lbi(lbi);
}

std::shared_ptr<XrtGraph> BuildXrtGraphEdges(
    std::shared_ptr<XrtGraph> &graph,
    const util::Map<const XrtNode *, util::Set<std::string>> &node_inputs,
    const util::Map<std::string, const XrtNode *> producers) {
  for (const auto &p : node_inputs) {
    const XrtNode *node = p.first;
    const util::Set<std::string> &inputs = p.second;
    for (const std::string &input : inputs) {
      const auto &it = producers.find(input);
      if (it != producers.end() && it->second != node) {
        XrtArgument argument(input);
        graph->Connect(it->second, node, argument);
      }
    }
  }
  return graph;
}

std::shared_ptr<XrtGraph> SetupXrtGraphEdges(
    std::shared_ptr<XrtGraph> &graph,
    const util::Map<const XrtNode *, const OpNode *> &op_nodes) {
  for (XrtEdge *edge : graph->Edges()) {
    const OpNode *src = op_nodes.at(edge->start());
    const OpNode *dst = op_nodes.at(edge->end());
    const std::string &name = edge->argument().name();
    // Set time shape
    std::vector<Shape> time_shape;
    time_shape.push_back(OutputTimeShape(src));
    time_shape.push_back(InputTimeShape(dst));
    edge->SetAttr<std::vector<Shape>>("time_shape", time_shape);
    // Set sbp policy
    std::vector<SbpParallel> sbp_policy;
    sbp_policy.push_back(BlobSbpPolicy(src, name));
    sbp_policy.push_back(BlobSbpPolicy(dst, name));
    edge->SetAttr<std::vector<SbpParallel>>("sbp_policy", sbp_policy);
  }
  return graph;
}

void SetupXrtNode(XrtNode *node, const OperatorConf &node_conf) {
  node->set_name(node_conf.name());
  node->set_type(ExtractOpTypeAsString(node_conf));
  node->set_backend(DeviceTypeToBackend(node_conf.device_type()));
}

void SetupXrtNode(XrtNode *node, const XrtLaunchOpConf::Argument &arg_conf) {
  node->set_name(arg_conf.name());
  node->set_type(_XrtArgumentOpType);
  node->set_backend(DeviceTypeToBackend(arg_conf.device_type()));
}

std::shared_ptr<XrtGraph> BuildXrtGraph(const XrtLaunchOpConf &launch_conf,
                                        const DeviceType &device_type,
                                        const JobDesc &job_desc) {
  std::shared_ptr<XrtGraph> graph(new XrtGraph);
  util::Map<std::string, const XrtNode *> producers;
  util::Map<const XrtNode *, util::Set<std::string>> node_inputs;

  for (const auto &arg_conf : launch_conf.attr().argument()) {
    XrtNode *node = graph->AddNode(arg_conf);
    SetupXrtNode(node, arg_conf);
    producers[arg_conf.out()] = node;
    node_inputs[node].insert(arg_conf.in());
  }

  for (const auto &node_conf : launch_conf.attr().node()) {
    XrtNode *node = graph->AddNode(node_conf);
    SetupXrtNode(node, node_conf);
    auto op = ConstructOp(node_conf, device_type, &job_desc);
    for (const std::string &bn : op->output_bns()) {
      std::string output = BlobIdToName(op->BnInOp2Lbi(bn));
      producers[output] = node;
    }
    for (const std::string &bn : op->input_bns()) {
      std::string input = BlobIdToName(op->BnInOp2Lbi(bn));
      node_inputs[node].insert(input);
    }
  }

  return BuildXrtGraphEdges(graph, node_inputs, producers);
}

std::shared_ptr<XrtGraph> BuildXrtGraph(const OpGraph *op_graph) {
  std::shared_ptr<XrtGraph> graph(new XrtGraph);
  util::Map<std::string, const XrtNode *> producers;
  util::Map<const XrtNode *, util::Set<std::string>> node_inputs;
  util::Map<const XrtNode *, const OpNode *> op_nodes;

  op_graph->TopoForEachNode([&](const OpNode *op_node) {
    const Operator *op = &op_node->op();
    XrtNode *node = graph->AddNode(op->op_conf());
    SetupXrtNode(node, op->op_conf());
    for (const std::string &bn : op->output_bns()) {
      std::string output = BlobIdToName(op->BnInOp2Lbi(bn));
      producers[output] = node;
    }
    for (const std::string &bn : op->input_bns()) {
      std::string input = BlobIdToName(op->BnInOp2Lbi(bn));
      node_inputs[node].insert(input);
    }
    op_nodes.emplace(node, op_node);
  });

  graph = BuildXrtGraphEdges(graph, node_inputs, producers);
  return SetupXrtGraphEdges(graph, op_nodes);
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
  return Parameter(const_cast<void *>(blob.dptr<void>()),
                   XrtShape(desc.shape(), desc.data_type()), name);
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

void ConvertXrtShapeToBlobDesc(const XrtShape &shape, BlobDesc *desc) {
  desc->mut_shape() = shape.shape();
  desc->set_data_type(shape.data_type());
}

}  // namespace xrt
}  // namespace oneflow
