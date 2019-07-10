#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/register/blob.h"
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

XlaCompiler::XlaCompiler(xla::LocalClient *client, xla::XlaBuilder *builder,
                         const XlaLaunchOpConf &launch_conf,
                         DeviceType device_type, ParallelContext parallel_ctx,
                         const std::vector<Blob *> &entry_blobs,
                         const std::vector<std::string> &entry_blob_names,
                         bool force_compile)
    : client_(client), builder_(builder), entry_names_(entry_blob_names),
      force_compile_(force_compile) {
  CHECK_EQ(entry_blobs.size(), entry_blob_names.size());

  std::unordered_map<std::string, BlobDesc> blob_descs;
  for (int i = 0; i < entry_blobs.size(); ++i) {
    const RtBlobDesc &rt_desc = entry_blobs[i]->blob_desc();
    BlobDesc blob_desc(rt_desc.shape(),
                       rt_desc.data_type(),
                       rt_desc.has_data_id_field(),
                       rt_desc.has_col_num_field(),
                       rt_desc.max_col_num());
    blob_descs.emplace(entry_blob_names[i], blob_desc);
  }
  graph_.reset(new XlaLaunchGraph(launch_conf, device_type));
  graph_->InferBlobDescs(&blob_descs, &parallel_ctx);

  for (const auto &pair : blob_descs) {
    LogicalBlobId lbi = BlobId(pair.first);
    arguments_.emplace(pair.first, Argument(lbi, pair.second));
  }
}

void XlaCompiler::BuildComputation(
    const std::unordered_map<Argument, XlaOprand> &entry_oprands,
    xla::Shape *output_shape, xla::XlaComputation *computation) {
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
    param.builder = builder_;

    SetupParamArguments(node, arguments_, &param);

    // Do compile and lower the operator computation to HLO instructions
    XlaOpContext op_context(param);
    compiler->Compile(&op_context);

    const auto &outputs = op_context.outputs();
    all_outputs.insert(outputs.begin(), outputs.end());
  });

  xla::StatusOr<xla::XlaComputation> computation_status = builder_->Build();
  CHECK(computation_status.ok());
  *computation = computation_status.ConsumeValueOrDie();
  // TODO(hjchen2) Remove debug logging
  VLOG(4) << computation->proto().DebugString();

  OF_CHECK_AND_ASSIGN(const auto& program_shape,
                      computation->GetProgramShape());
  *output_shape = program_shape.result();
}

void XlaCompiler::BuildExecutable(
    const CompilationResult &result,
    std::unique_ptr<xla::LocalExecutable> *executable) {
  std::vector<const xla::Shape*> argument_layouts(
      result.xla_input_shapes.size());
  for (int i = 0; i < result.xla_input_shapes.size(); ++i) {
    argument_layouts[i] = &result.xla_input_shapes[i];
  }

  xla::ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(client_->default_device_ordinal());
  build_options.set_result_layout(result.xla_output_shape);

  OF_CHECK_AND_ASSIGN(
      *executable,
      client_->Compile(result.computation, argument_layouts, build_options));
  DLOG(INFO) << "BuildExecutable done";
}

CompilationResult XlaCompiler::Compile() {
  CHECK_NOTNULL(graph_);
  
  CompilationResult result;
  std::unordered_map<Argument, XlaOprand> entry_oprands;
  SetupEntryOprands(&entry_oprands, &result.xla_input_shapes);

  BuildComputation(entry_oprands, &result.xla_output_shape,
                   &result.computation);

  BuildExecutable(result, &result.executable);

  return std::move(result);
}

void XlaCompiler::SetupEntryOprands(
    std::unordered_map<Argument, XlaOprand> *entry_oprands,
    std::vector<xla::Shape> *input_shapes) {
  for (int i = 0; i < entry_names_.size(); ++i) {
    Argument arg = arguments_.at(entry_names_[i]);
    xla::Shape shape = OfShapeToXlaShape(arg.shape(), arg.data_type());
    input_shapes->push_back(shape);

    // Treat all inputs as xla Parameters
    xla::XlaOp handle = xla::Parameter(builder_, i, shape,
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
