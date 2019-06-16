#include "glog/logging.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/job/resource.pb.h"  // DataType
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_op_compiler_registry.h"
#include "oneflow/core/compiler/of2xla/xla_op_context.h"
#include "oneflow/core/compiler/of2xla/xla_op_compiler.h"
#include "oneflow/core/compiler/of2xla/xla_graph_compiler.h"

namespace oneflow {
namespace mola {

static XlaOpCompiler *CreateXlaOpCompiler(
    const std::string &backend, const std::string &op_type) {
  return XlaOpCompilerRegistry::Build(backend)[op_type]();
}

void XlaGraphCompiler::Compile() {
  CHECK_NOTNULL(graph_);
  // Collect all operators outputs
  std::unordered_map<LogicalBlobId, XlaOprand> all_outputs;

  graph_->TopoForEachNode([&](OpNode *node) -> void {
    // Generate backend from the node operator's device type
    std::string backend = [&]() -> std::string {
      const DeviceType device_type = node->op().device_type();
      switch (device_type) {
        case DeviceType::kCPU:
          return "CPU";
        case DeviceType::kGPU:
          return "CUDA";
        default:
          LOG(ERROR) << "Not supported DeviceType (" << device_type
                     << ") in XlaGraphCompiler::Compile";
          return NoneString;
      }
    }();

    std::string type = ExtractOpTypeAsString(node->op());
    if (!IsOpTypeCompiled(backend, type)) {
      return;
    }
    // Create operator compiler
    XlaOpCompiler *compiler = CreateXlaOpCompiler(backend, type);

    // Setup input oprands from outputs of previous nodes
    std::unordered_map<LogicalBlobId, XlaOprand> input_oprands;
    for (const OpEdge *edge : node->in_edges()) {
      for (const LogicalBlobId &blob_id : edge->lbis()) {
        CHECK_GT(all_outputs.count(blob_id), 0);
        input_oprands.emplace(blob_id, all_outputs[blob_id]);
      }
    }

    std::unordered_map<LogicalBlobId, DataType> output_types;
    const auto &output_edges = node->out_edges();
    for (const OpEdge *edge : output_edges) {
      for (const LogicalBlobId &blob_id : edge->lbis()) {
        const BlobDesc &blob_desc = node->LogicalBlobDesc4Lbi(blob_id);
        output_types.emplace(blob_id, blob_desc.data_type());
      }
    }
   
    // Setup XlaOpContext Param to build a XlaOpContext
    XlaOpContext::Param param;
    param.inputs = std::move(input_oprands);
    param.output_types = std::move(output_types);
    param.num_outputs = output_types.size();
    param.op_conf = &(node->op().GetCustomizedConf());
    param.builder = builder_;

    param.blob_id_from_string_fn = [&](const std::string &name) -> LogicalBlobId {
      const Operator &op = node->op();
      return op.BnInOp2Lbi(name);
    };

    // Do compile and lower the graph computation to HLO instructions
    XlaOpContext op_context(param);
    compiler->Compile(&op_context);

    for (const auto &kv_pair : op_context.OutputOprands()) {
      all_outputs.emplace(kv_pair);
    }
  });
}

}  // namespace mola
}  // namespace oneflow
