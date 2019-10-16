#include "oneflow/xla/of2xla/xla_graph_compiler.h"
#include "absl/strings/str_cat.h"
#include "glog/logging.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/xla/of2xla/xla_argument.h"
#include "oneflow/xla/of2xla/xla_graph.h"
#include "oneflow/xla/of2xla/xla_op_compiler.h"
#include "oneflow/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/xla/of2xla/xla_op_context.h"
#include "oneflow/xla/of2xla/xla_shape.h"
#include "oneflow/xla/of2xla/xla_utility.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/service.h"

namespace oneflow {
namespace mola {

XlaGraphCompiler::XlaGraphCompiler(xla::LocalClient *client,
                                   xla::XlaBuilder *builder)
    : client_(client), builder_(builder) {}

void XlaGraphCompiler::BuildComputation(
    const XlaGraph *graph,
    const std::unordered_map<Argument, XlaOprand> &entry_oprands,
    const std::vector<Argument> &return_arguments, xla::Shape *output_shape,
    xla::XlaComputation *computation) {
  // All operator's output oprands collector
  std::unordered_map<Argument, XlaOprand> all_outputs(entry_oprands);

  TopologyVisit(*graph, [&](const XlaNode *node) {
    const std::string &backend = node->backend();
    const std::string &op_type = node->op_type();

    // Setup input oprands from outputs of previous nodes
    std::unordered_map<Argument, XlaOprand> input_oprands;
    for (const std::string &in : node->input_bns()) {
      std::string blob_name = BlobName(node->Input(in));
      Argument argument = arguments_.at(blob_name);
      XlaOprand oprand = all_outputs.at(argument);
      input_oprands.emplace(argument, oprand);
    }

    xla::OpMetadata metadata;
    metadata.set_op_type(op_type);
    metadata.set_op_name(node->op_name());
    builder_->SetOpMetadata(metadata);

    // Setup XlaOpContext Param to build a XlaOpContext
    XlaOpContext::Param param;
    param.backend = backend;
    param.inputs = std::move(input_oprands);
    param.op_conf = &node->proto_conf();
    param.builder = builder_;
    param.num_outputs = node->output_bns().size();

    SetupNodeArguments(node, arguments_, &param);

    // Do compile and lower the operator computation to HLO instructions
    auto op_compiler = CreateXlaOpCompiler(backend, op_type);
    XlaOpContext op_context(param);
    op_compiler->Compile(&op_context);

    builder_->ClearOpMetadata();

    // Always insert new output into `all_outputs`
    const auto &outputs = op_context.outputs();
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
      all_outputs[it->first] = it->second;
    }
  });

  // Always insert a final tuple XlaOp to ensure the computation ends with
  // all the return values. This also make sure that it returns a tuple shape
  // after runing the executable
  std::vector<xla::XlaOp> return_vals(return_arguments.size());
  for (int i = 0; i < return_vals.size(); ++i) {
    const Argument &arg = return_arguments[i];
    return_vals[i] = all_outputs.at(arg).AsXlaOp(builder_);
  }
  xla::Tuple(builder_, return_vals);

  xla::StatusOr<xla::XlaComputation> computation_status = builder_->Build();
  CHECK(computation_status.ok());
  *computation = computation_status.ConsumeValueOrDie();
  // TODO(hjchen2) Remove debug logging
  VLOG(4) << computation->proto().DebugString();

  OF_CHECK_AND_ASSIGN(const auto &program_shape,
                      computation->GetProgramShape());
  *output_shape = program_shape.result();
  for (int i = 0; i < return_vals.size(); ++i) {
    xla::Shape *output_sub_shape =
        xla::ShapeUtil::GetMutableSubshape(output_shape, {i});
    xla::LayoutUtil::SetToDefaultLayout(output_sub_shape);
  }
}

void XlaGraphCompiler::BuildExecutable(
    const CompilationResult &result,
    std::unique_ptr<xla::LocalExecutable> *executable) {
  std::vector<const xla::Shape *> argument_layouts(
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

CompilationResult XlaGraphCompiler::Compile(
    const XlaGraph *graph, const std::vector<Argument> &entry_arguments,
    const std::vector<Argument> &return_arguments,
    const std::vector<std::string> &entry_names,
    const std::vector<std::string> &return_names,
    const std::vector<xla::XlaBuilder::InputOutputAlias> &aliases) {
  CHECK_NOTNULL(graph);
  CompilationResult result;
  for (const xla::XlaBuilder::InputOutputAlias &alias : aliases) {
    builder_->SetUpAlias(alias.output_index, alias.param_number,
                         alias.param_index);
  }
  BuildCompilationArguments(graph, entry_arguments, return_arguments,
                            entry_names, return_names);
  std::unordered_map<Argument, XlaOprand> entry_oprands;
  BuildEntryParameters(entry_names, &entry_oprands, &result.xla_input_shapes);

  BuildComputation(graph, entry_oprands, return_arguments,
                   &result.xla_output_shape, &result.computation);

  BuildExecutable(result, &result.executable);

  return std::move(result);
}

void XlaGraphCompiler::BuildEntryParameters(
    const std::vector<std::string> &entry_names,
    std::unordered_map<Argument, XlaOprand> *entry_oprands,
    std::vector<xla::Shape> *input_shapes) {
  for (int i = 0; i < entry_names.size(); ++i) {
    Argument arg = arguments_.at(entry_names[i]);
    xla::Shape shape = OfShapeToXlaShape(arg.shape(), arg.data_type());
    input_shapes->push_back(shape);

    // Treat any input as xla Parameter
    xla::XlaOp handle =
        xla::Parameter(builder_, i, shape, absl::StrCat("arg", i));
    entry_oprands->emplace(arg, XlaOprand::XlaOp(handle));
  }
}

void XlaGraphCompiler::SetupNodeArguments(
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

void XlaGraphCompiler::BuildCompilationArguments(
    const XlaGraph *graph, const std::vector<Argument> &entry_arguments,
    const std::vector<Argument> &return_arguments,
    const std::vector<std::string> &entry_names,
    const std::vector<std::string> &return_names) {
  CHECK_EQ(entry_arguments.size(), entry_names.size());
  CHECK_EQ(return_arguments.size(), return_names.size());
  for (int i = 0; i < entry_arguments.size(); ++i) {
    arguments_.emplace(entry_names[i], entry_arguments[i]);
  }
  for (int i = 0; i < return_arguments.size(); ++i) {
    arguments_.emplace(return_names[i], return_arguments[i]);
  }

  const std::vector<Argument> arguments = graph->Arguments();
  for (const Argument &argument : arguments) {
    arguments_.emplace(argument.blob_name(), argument);
  }
}

}  // namespace mola
}  // namespace oneflow
