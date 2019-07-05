#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_shape.h"
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

CompileContext::CompileContext(
    const XlaGraph *graph, xla::XlaBuilder *builder,
    const std::vector<std::string> &input_arg_names,
    const std::unordered_map<std::string, BlobDesc> &infered_blob_descs,
    bool force_compile) : graph_(graph), builder_(builder),
                          force_compile_(force_compile) {
  for (const auto &pair : infered_blob_descs) {
    LogicalBlobId lbi = BlobId(pair.first);
    arguments_.emplace(lbi, Argument(lbi, pair.second));
  }
  // Build input XlaOprands
  BuildArgumentOprands(input_arg_names);
}

void CompileContext::BuildArgumentOprands(
    const std::vector<std::string> &input_arg_names) {
  for (int i = 0; i < input_arg_names.size(); ++i) {
    LogicalBlobId lbi = BlobId(input_arg_names[i]);
    Argument argument = arguments_.at(lbi);
    xla::Shape shape = OfShapeToXlaShape(argument.shape(),
                                         argument.data_type());
    xla::XlaOp handle = xla::Parameter(builder_, i, shape,
                                       absl::StrCat("arg", i));
    input_oprands_.emplace(argument, XlaOprand::XlaOp(handle));
  }
}

void XlaGraphCompiler::Compile(CompileContext *ctx) {
  CHECK_NOTNULL(ctx->graph_);
  // All operator's output oprands collector
  std::unordered_map<Argument, XlaOprand> all_outputs(ctx->input_oprands_);

  TopologyVisit(*ctx->graph_, [&](const XlaNode *node) {
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
    param.builder = ctx->builder_;

    SetupParamArguments(node, ctx->arguments_, &param);

    // Do compile and lower the operator computation to HLO instructions
    XlaOpContext op_context(param);
    compiler->Compile(&op_context);

    const auto &outputs = op_context.outputs();
    all_outputs.insert(outputs.begin(), outputs.end());
  });
}

void XlaGraphCompiler::SetupParamArguments(
                const XlaNode *node,
                const std::unordered_map<LogicalBlobId, Argument> &arguments,
                XlaOpContext::Param *param) {
  std::unordered_map<std::string, Argument> op_arguments;
  for (const std::string &in : node->input_bns()) {
    LogicalBlobId lbi = node->Input(in);
    op_arguments.emplace(in, arguments.at(lbi));
  }
  for (const std::string &out : node->output_bns()) {
    LogicalBlobId lbi = node->Output(out);
    op_arguments.emplace(out, arguments.at(lbi));
  }
  param->arguments = std::move(op_arguments);
}

}  // namespace mola
}  // namespace oneflow
