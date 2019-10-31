#include "oneflow/xrt/xla/xla_graph_compiler.h"
#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/ops/op_kernel.h"
#include "oneflow/xrt/xla/xla_resource_manager.h"
#include "oneflow/xrt/xla/xla_shape.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace oneflow {
namespace xrt {
namespace mola {

Argument XlaGraphCompiler::ArgFromParameter(const Parameter &param) {
  return Argument(param.name(), param.shape(), param.data_type());
}

void XlaGraphCompiler::SetupContextParam(
    const XrtNode *node, const util::Map<Argument, Operand> &all_outputs,
    OpContext::Param *context_param) {
  const PbMessage *message = &node->param();
  if (!node->IsArgumentNode()) {
    util::GetOneofMessage(node->param(), "op_type", &message);
    CHECK(message) << "Cann't get op_type message in node param.";
  }

  util::Map<Argument, Operand> input_ops;
  util::Map<std::string, Argument> input_output_args;
  for (const XrtEdge *edge : node->in_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument &arg = edge->argument();
      const Operand &operand = all_outputs.at(arg);
      input_ops.emplace(arg, operand);
      const std::string &k = arg.meta_data().consume_key();
      input_output_args.emplace(k, arg);
    }
  }
  for (const XrtEdge *edge : node->out_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument &arg = edge->argument();
      const std::string &k = arg.meta_data().produce_key();
      input_output_args.emplace(k, arg);
    }
  }

  if (node->IsInArgumentNode()) {
    std::string input = util::GetAttrAsString(*message, "in");
    const Argument &arg = arguments_.at(input);
    input_ops.emplace(arg, entry_operands_.at(arg));
    input_output_args.emplace("in", arg);
  }
  if (node->IsOutArgumentNode()) {
    std::string output = util::GetAttrAsString(*message, "out");
    const Argument &arg = arguments_.at(output);
    input_output_args.emplace("out", arg);
  }

  size_t num_outputs = input_output_args.size() - input_ops.size();
  CHECK_GE(num_outputs, 0) << "Output num should not less than 0.";
  context_param->device = node->device();
  context_param->builder = builder_.get();
  context_param->message = message;
  context_param->arguments = std::move(input_output_args);
  context_param->inputs = std::move(input_ops);
  context_param->num_outputs = num_outputs;
}

void XlaGraphCompiler::BuildComputation(
    const XrtGraph *graph, const std::vector<Argument> &return_args,
    xla::Shape *output_shape, xla::XlaComputation *computation) {
  // All operator's output operands.
  util::Map<Argument, Operand> all_outputs;
  // Compile each node as topology order.
  algorithm::TopologyVisit(*graph, [&](const XrtNode *node) {
    // Setup param to build an OpContext.
    OpContext::Param param;
    SetupContextParam(node, all_outputs, &param);
    if (use_meta_data_) {
      xla::OpMetadata metadata;
      metadata.set_op_type(node->type());
      metadata.set_op_name(node->name());
      builder_->SetOpMetadata(metadata);
    }
    // Do compile, lower the operator computation to HLO instructions.
    auto op_kernel = BuildOpKernel(node->device(), node->type());
    OpContext op_context(param);
    op_kernel->Compile(&op_context);

    if (use_meta_data_) {
      builder_->ClearOpMetadata();
    }
    // Always insert the new output into `all_outputs`.
    const auto &outputs = op_context.outputs();
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
      all_outputs[it->first] = it->second;
    }
  });

  // Always insert a final tuple XlaOp to ensure the computation ends with
  // all the return values. This also make sure that it returns a tuple shape
  // after runing the executable.
  std::vector<xla::XlaOp> return_vals(return_args.size());
  for (int i = 0; i < return_vals.size(); ++i) {
    const Argument &arg = return_args[i];
    return_vals[i] = all_outputs.at(arg).AsXlaOp(builder_.get());
  }
  xla::Tuple(builder_.get(), return_vals);

  xla::StatusOr<xla::XlaComputation> computation_status = builder_->Build();
  CHECK(computation_status.ok());
  *computation = computation_status.ConsumeValueOrDie();
  // TODO(hjchen2) Remove debug logging.
  VLOG(4) << computation->proto().DebugString();

  MOLA_CHECK_AND_ASSIGN(const auto &program_shape,
                        computation->GetProgramShape());
  *output_shape = program_shape.result();
  for (int i = 0; i < return_vals.size(); ++i) {
    xla::Shape *output_sub_shape =
        xla::ShapeUtil::GetMutableSubshape(output_shape, {i});
    xla::LayoutUtil::SetToDefaultLayout(output_sub_shape);
  }
}

std::shared_ptr<Executable> XlaGraphCompiler::BuildExecutable(
    const std::vector<xla::Shape> &xla_input_shapes,  // NOLINT
    const xla::Shape &xla_output_shape,               // NOLINT
    const xla::XlaComputation &computation) {
  std::vector<const xla::Shape *> argument_layouts(xla_input_shapes.size());
  for (int i = 0; i < xla_input_shapes.size(); ++i) {
    argument_layouts[i] = &xla_input_shapes[i];
  }

  xla::LocalClient *client =
      resource_mgr::GetOrCreateLocalClient(this->device_);

  xla::ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(this->device_ordinal_);
  build_options.set_result_layout(xla_output_shape);
  MOLA_CHECK_AND_ASSIGN(
      auto executable,
      client->Compile(computation, argument_layouts, build_options));
  return std::make_shared<XlaExecutable>(xla_input_shapes, xla_output_shape,
                                         executable);
}

void XlaGraphCompiler::BuildEntryParameters(
    const std::vector<Parameter> &entry_params,
    std::vector<xla::Shape> *input_shapes) {
  for (int i = 0; i < entry_params.size(); ++i) {
    const DataType data_type = entry_params[i].data_type();
    const Shape &shape = entry_params[i].shape();
    xla::Shape xla_shape = OfShapeToXlaShape(shape, data_type);
    input_shapes->push_back(xla_shape);
    // Treat all inputs as xla parameters.
    xla::XlaOp handle =
        xla::Parameter(builder_.get(), i, xla_shape, absl::StrCat("arg", i));
    Argument arg = ArgFromParameter(entry_params[i]);
    arguments_.emplace(entry_params[i].name(), arg);
    entry_operands_.emplace(arg, Operand::XlaOp(handle));
  }
}

xla::ShapeIndex MakeShapeIndex(const std::vector<int> &shape) {
  xla::ShapeIndex shape_index;
  for (int i = 0; i < shape.size(); ++i) {
    shape_index.push_back(shape[i]);
  }
  return std::move(shape_index);
}

std::shared_ptr<Executable> XlaGraphCompiler::Compile(
    const XrtGraph *graph, const std::vector<Parameter> &entry_params,
    const std::vector<Parameter> &return_params,
    const std::vector<InputOutputAlias> &aliases) {
  builder_ = std::make_shared<xla::XlaBuilder>(
      absl::StrCat("XlaBuilder_", this->name_));
  for (const InputOutputAlias &alias : aliases) {
    builder_->SetUpAlias(MakeShapeIndex(alias.output_index()),
                         alias.param_number(),
                         MakeShapeIndex(alias.param_index()));
  }
  std::vector<Argument> return_args(return_params.size());
  for (int i = 0; i < return_params.size(); ++i) {
    return_args[i] = ArgFromParameter(return_params[i]);
    arguments_.emplace(return_params[i].name(), return_args[i]);
  }
  std::vector<xla::Shape> input_shapes;
  xla::Shape output_shape;
  xla::XlaComputation computation;
  BuildEntryParameters(entry_params, &input_shapes);
  BuildComputation(graph, return_args, &output_shape, &computation);

  return BuildExecutable(input_shapes, output_shape, computation);
}

REGISTER_GRAPH_COMPILER(XrtEngine::XLA, XlaGraphCompiler);

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
