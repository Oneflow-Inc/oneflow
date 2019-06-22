#include "glog/logging.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/job/resource.pb.h"  // DataType
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_op_compiler_registry.h"
#include "oneflow/core/compiler/of2xla/xla_argument.h"
#include "oneflow/core/compiler/of2xla/xla_op_context.h"
#include "oneflow/core/compiler/of2xla/xla_op_compiler.h"
#include "oneflow/core/compiler/of2xla/xla_graph.h"
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
  std::unordered_map<Argument, XlaOprand> all_outputs;

  for (auto &node : graph_->Nodes()) {
    const std::string &backend = node->backend();
    const std::string &op_type = node->op_type();
    // Create operator compiler
    XlaOpCompiler *compiler = CreateXlaOpCompiler(backend, op_type);

    // Setup input oprands from outputs of previous nodes
    std::unordered_map<Argument, XlaOprand> input_oprands;
    for (const XlaEdge *edge : node->in_edges()) {
      const Argument &arg = edge->argument();
      CHECK_GT(all_outputs.count(arg), 0);
      input_oprands.emplace(arg, all_outputs[arg]);
    }

    // Setup XlaOpContext Param to build a XlaOpContext
    XlaOpContext::Param param;
    param.inputs = std::move(input_oprands);
    param.op_conf = &node->proto_conf();
    param.builder = builder_;

    param.argument_from_string_fn = [&](const std::string &name) -> Argument {
      const Operator *op = node->op();
      const LogicalBlobId &lbi = op->BnInOp2Lbi(name);
      return Argument(lbi, node->node()->LogicalBlobDesc4Lbi(lbi));
    };

    // Do compile and lower the graph computation to HLO instructions
    XlaOpContext op_context(param);
    compiler->Compile(&op_context);

    const auto &outputs = op_context.OutputOprands();
    all_outputs.insert(outputs.begin(), outputs.end());
  }
}

}  // namespace mola
}  // namespace oneflow
