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
#include "oneflow/core/compiler/of2xla/xla_compiler.h"

namespace oneflow {
namespace mola {

static XlaOpCompiler *CreateXlaOpCompiler(
    const std::string &backend, const std::string &op_type) {
  return XlaOpCompilerRegistry::Build(backend)[op_type]();
}

XlaCompiler::XlaCompiler(
    const OperatorConf &op_conf, DeviceType device_type,
    ParallelContext parallel_ctx,
    const std::unordered_map<std::string, BlobDesc> &entry_blob_descs,
    bool force_compile) : force_compile_(force_compile) {
  entry_names_.resize(entry_blob_descs.size()); 
  std::transform(entry_blob_descs.begin(), entry_blob_descs.end(),
                 entry_names_.begin(),
                 [](const std::pair<const std::string, BlobDesc> &p) {
                   return p.first;
                 });

  std::unordered_map<std::string, BlobDesc> blob_descs(entry_blob_descs);
  CHECK(op_conf.has_xla_launch_conf());
  graph_.reset(new XlaLaunchGraph(op_conf.xla_launch_conf(), device_type));
  graph_->InferBlobDescs(&blob_descs, &parallel_ctx);

  for (const auto &pair : blob_descs) {
    LogicalBlobId lbi = BlobId(pair.first);
    arguments_.emplace(pair.first, Argument(lbi, pair.second));
  }

  builder_.reset(new xla::XlaBuilder(op_conf.name()));
}

void XlaCompiler::BuildComputation(
    const std::unordered_map<Argument, XlaOprand> &entry_oprands) {
  // All operator's output oprands collector
  std::unordered_map<Argument, XlaOprand> all_outputs;

  TopologyVisit(*graph_, [&](const XlaNode *node) {
    const std::string &backend = node->backend();
    const std::string &op_type = node->op_type();
    // Create operator compiler
    XlaOpCompiler *compiler = CreateXlaOpCompiler(backend, op_type);

    // Setup input oprands from outputs of previous nodes
    auto input_oprands = entry_oprands;
    for (const XlaEdge *edge : node->in_edges()) {
      const Argument &arg = edge->argument();
      CHECK_GT(all_outputs.count(arg), 0);
      input_oprands.emplace(arg, all_outputs[arg]);
    }

    // Setup XlaOpContext Param to build a XlaOpContext
    XlaOpContext::Param param;
    param.inputs = std::move(input_oprands);
    param.op_conf = &node->proto_conf();
    param.builder = builder_.get();

    SetupParamArguments(node, arguments_, &param);

    // Do compile and lower the operator computation to HLO instructions
    XlaOpContext op_context(param);
    compiler->Compile(&op_context);

    const auto &outputs = op_context.outputs();
    all_outputs.insert(outputs.begin(), outputs.end());
  });
}

void XlaCompiler::BuildExecutable() {
  // Build xla computation (just for debug)
  xla::StatusOr<xla::XlaComputation> computation_status = builder_->Build();
  xla::XlaComputation computation = computation_status.ConsumeValueOrDie();
  xla::StatusOr<xla::ProgramShape> program_shape_status = computation.GetProgramShape();
  xla::ProgramShape program_shape = program_shape_status.ConsumeValueOrDie();
  DLOG(INFO) << computation.proto().DebugString();
}

void XlaCompiler::Compile() {
  CHECK_NOTNULL(graph_);

  std::unordered_map<Argument, XlaOprand> entry_oprands;
  SetupEntryOprands(&entry_oprands);

  BuildComputation(entry_oprands);

  BuildExecutable();
}

void XlaCompiler::SetupEntryOprands(
    std::unordered_map<Argument, XlaOprand> *entry_oprands) {
  for (int i = 0; i < entry_names_.size(); ++i) {
    Argument arg = arguments_.at(entry_names_[i]);
    xla::Shape shape = OfShapeToXlaShape(arg.shape(), arg.data_type());
    xla::XlaOp handle = xla::Parameter(builder_.get(), i, shape,
                                       absl::StrCat("arg", i));
    entry_oprands->emplace(arg, XlaOprand::XlaOp(handle));
  }
}

void XlaCompiler::SetupParamArguments(
    const XlaNode *node,
    const std::unordered_map<std::string, Argument> &arguments,
    XlaOpContext::Param *param) {
  std::unordered_map<std::string, Argument> op_arguments;
  for (const std::string &in : node->input_bns()) {
    std::string arg_name = BlobName(node->Input(in));
    op_arguments.emplace(in, arguments.at(arg_name));
  }
  for (const std::string &out : node->output_bns()) {
    std::string arg_name = BlobName(node->Output(out));
    op_arguments.emplace(out, arguments.at(arg_name));
  }
  param->arguments = std::move(op_arguments);
}

}  // namespace mola
}  // namespace oneflow
