#include "glog/logging.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/compiler/of2xla/xla_graph_compiler.h"

namespace oneflow {
namespace mola {

static XlaOpCompiler *CreateXlaOpCompiler(
    const std::string &backend, const std::string &op_type) {
  return XlaOpCompilerRegistry::Get(backend)[op_type]();
}

void XlaGraphCompiler::Compile() {
  CHECK_NOTNULL(graph_);
  // TODO(hjchen2): Init all input XlaOp

  // Collect all operators's outputs
  std::unordered_map<std::string, std::vector<XlaOp>> all_outputs;

  graph_->TopoForEachNode([&](OpNode *node) -> void {
    // TODO(hjchen2): node->op().device_type()
    std::string backend = "CPU";
    std::string typpe = "";  // TODO
    std::string op_name = node->op().op_name();
    // Create operator compiler
    XlaOpCompiler *compiler = CreateXlaOpCompiler(backend, type);

    // Create compiler context
    XlaOpContext::Param param;
//    param.inputs =
//    param.input_types = 
//    param.output_types = 
//    param.num_outputs = 
    param.op_conf = node->op().GetCustomizedConf();
    param.builder = builder_;

    XlaOpContext op_context(param);
    compiler->Compile(&op_context);

    all_outputs[op_name] = std::move(op_context.OutputOprands());
  });
}

}  // namespace mola
}  // namespace oneflow
