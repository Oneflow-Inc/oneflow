#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/engine/xla/of2xla/xla_utility.h"
#include "oneflow/engine/xla/of2xla/xla_shape.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler_registry.h"
#include "oneflow/engine/xla/of2xla/xla_argument.h"
#include "oneflow/engine/xla/of2xla/xla_op_context.h"
#include "oneflow/engine/xla/of2xla/xla_op_compiler.h"
#include "oneflow/engine/xla/of2xla/xla_graph.h"
#include "oneflow/engine/xla/of2xla/xla_graph_compiler.h"

namespace oneflow {
namespace mola {

XlaGraphCompiler::XlaGraphCompiler(
    xla::LocalClient *client, xla::XlaBuilder *builder, XlaGraph *graph,
    ParallelContext parallel_ctx, const std::vector<Blob *> &entry_blobs,
    const std::vector<std::string> &entry_blob_names,
    const std::vector<std::string> &return_blob_names,
    const bool alias_input_output)
    : client_(client), builder_(builder), graph_(graph),
      entry_names_(entry_blob_names), return_names_(return_blob_names),
      alias_input_output_(alias_input_output) {
  const std::vector<Argument> arguments = graph_->Arguments();
  for (const Argument &argument : arguments) {
    arguments_.emplace(argument.blob_name(), argument);
  }
  CHECK_EQ(entry_blobs.size(), entry_blob_names.size());

  std::unordered_map<std::string, BlobDesc> blob_descs;
  for (int i = 0; i < entry_blobs.size(); ++i) {
    const RtBlobDesc &runtime_desc = entry_blobs[i]->blob_desc();
    BlobDesc blob_desc(runtime_desc.shape(),
                       runtime_desc.data_type(),
                       runtime_desc.has_data_id_field(),
                       runtime_desc.has_col_num_field(),
                       runtime_desc.max_col_num());
    LogicalBlobId blob_id = BlobId(entry_blob_names[i]);
    // TODO(hjchen2): Check blob shape and data type if existed
    arguments_.emplace(entry_blob_names[i], Argument(blob_id, blob_desc));
  }
}

void XlaGraphCompiler::BuildComputation(
    const std::unordered_map<Argument, XlaOprand> &entry_oprands,
    const std::vector<Argument> &return_arguments,
    xla::Shape *output_shape, xla::XlaComputation *computation) {
  // All operator's output oprands collector
  std::unordered_map<Argument, XlaOprand> all_outputs(entry_oprands);

  TopologyVisit(*graph_, [&](const XlaNode *node) {
    const std::string &backend = node->backend();
    const std::string &op_type = node->op_type();
    // Create operator compiler
    auto op_compiler = CreateXlaOpCompiler(backend, op_type);

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

    SetupParamArguments(node, arguments_, &param);

    // Do compile and lower the operator computation to HLO instructions
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

  OF_CHECK_AND_ASSIGN(const auto& program_shape,
                      computation->GetProgramShape());
  *output_shape = program_shape.result();
  for (int i = 0; i < return_vals.size(); ++i) {
    xla::Shape* output_sub_shape = xla::ShapeUtil::GetMutableSubshape(
        output_shape, {i});
    xla::LayoutUtil::SetToDefaultLayout(output_sub_shape);
  }
}

void XlaGraphCompiler::BuildExecutable(
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

CompilationResult XlaGraphCompiler::Compile() {
  CHECK_NOTNULL(graph_);
  CompilationResult result;
  result.alias_input_output = alias_input_output_;

  std::vector<std::string> input_names = entry_names_;
  const int return_size = return_names_.size();
  int argument_index = entry_names_.size();
  std::vector<Argument> return_arguments(return_size);
  for (int i = 0; i < return_size; ++i, ++argument_index) {
    // Alias outputs to input arguments to update in-place
    if (alias_input_output_) {
      builder_->SetUpAlias({i}/*output_index*/, argument_index/*param_number*/,
                           {}/*param_index*/);
      input_names.push_back(return_names_[i]);
    }
    return_arguments[i] = arguments_.at(return_names_[i]);
  }
  std::unordered_map<Argument, XlaOprand> entry_oprands;
  SetupEntryOprands(input_names, &entry_oprands, &result.xla_input_shapes);

  BuildComputation(entry_oprands, return_arguments, &result.xla_output_shape,
                   &result.computation);

  BuildExecutable(result, &result.executable);
  
  return std::move(result);
}

void XlaGraphCompiler::SetupEntryOprands(
    const std::vector<std::string> &entry_names,
    std::unordered_map<Argument, XlaOprand> *entry_oprands,
    std::vector<xla::Shape> *input_shapes) {
  for (int i = 0; i < entry_names.size(); ++i) {
    Argument arg = arguments_.at(entry_names[i]);
    xla::Shape shape = OfShapeToXlaShape(arg.shape(), arg.data_type());
    input_shapes->push_back(shape);

    // Treat any input as xla Parameter
    xla::XlaOp handle = xla::Parameter(builder_, i, shape,
                                       absl::StrCat("arg", i));
    entry_oprands->emplace(arg, XlaOprand::XlaOp(handle));
  }
}

void XlaGraphCompiler::SetupParamArguments(
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
