#include "oneflow/xrt/xla/xla_graph_compiler.h"

namespace oneflow {
namespace xrt {
namespace mola {

/*
void XlaGraphCompiler::BuildComputation(
    const XrtGraph *graph, const std::vector<Argument> &return_params,
    xla::Shape *output_shape, xla::XlaComputation *computation) {
  // All operator's output oprands collector
  util::Map<Argument, XlaOprand> all_outputs(entry_oprands_);

  algorithm::TopologyVisit(*graph, [&](const XrtNode *node) {
    // Setup input oprands from outputs of previous nodes
    util::Map<Argument, XlaOprand> input_oprands;
    if (node->IsInArgumentNode()) {

    }

    for (const std::string &in : node->input_bns()) {
      std::string blob_name = BlobName(node->Input(in));
      Argument argument = arguments_.at(blob_name);
      XlaOprand oprand = all_outputs.at(argument);
      input_oprands.emplace(argument, oprand);
    }

    const std::string &type = node->type();
    const std::string &backend = node->backend();
    xla::OpMetadata metadata;
    metadata.set_op_type(type);
    metadata.set_op_name(node->name());
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
    auto op_compiler = CreateXlaOpCompiler(backend, type);
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
  std::vector<xla::XlaOp> return_vals(return_params.size());
  for (int i = 0; i < return_vals.size(); ++i) {
    const Argument &arg = return_params[i];
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
}

void XlaGraphCompiler::BuildEntryParameters(
    const std::vector<Argument> &entry_params,
    std::vector<xla::Shape> *input_shapes) {
  for (int i = 0; i < entry_params.size(); ++i) {
    const Argument &arg = entry_params[i];
    xla::Shape shape = OfShapeToXlaShape(arg.shape(), arg.data_type());
    input_shapes->push_back(shape);
    // Treat all inputs as xla parameters.
    xla::XlaOp handle =
        xla::Parameter(builder_, i, shape, absl::StrCat("arg", i));
    entry_oprands_->emplace(arg, XlaOprand::XlaOp(handle));
  }
}
std::shared_ptr<Executable> XlaGraphCompiler::Compile(
    const XrtGraph *graph, const std::vector<Argument> &entry_params,
    const std::vector<Argument> &return_params,
    const std::vector<InputOutputAlias> &aliases) {
  for (const InputOutputAlias &alias : aliases) {
    builder_->SetUpAlias(alias.output_index, alias.param_number,
                         alias.param_index);
  }
  std::vector<xla::Shape> input_shapes;
  xla::Shape output_shape;
  BuildEntryParameters(entry_params, &input_shapes);

  xla::XlaComputation computation;
  BuildComputation(graph, return_params, &output_shape, &computation);

  return BuildExecutable(input_shapes, output_shape, computation);
}
*/

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
