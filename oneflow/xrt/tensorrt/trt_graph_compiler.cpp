#include "oneflow/xrt/tensorrt/trt_graph_compiler.h"
#include "oneflow/xrt/node_util.h"
#include "oneflow/xrt/tensorrt/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

void TrtGraphCompiler::PopulateEntryParams(const std::vector<Parameter> &entry_params) {
  for (const Parameter &param : entry_params) {
    Argument arg = ArgFromParameter(param);
    TrtValue value = TrtValue::Parameter(builder_.get(), param);
    operands_[arg] = std::move(value);
  }
}

Argument TrtGraphCompiler::ArgFromParameter(const Parameter &param) {
  return Argument(param.name(), param.shape(), param.data_type());
}

void TrtGraphCompiler::SetupKernelContextParam(const XrtNode *node,
                                               TrtOpContext::Param *context_param) {
  util::Map<Argument, TrtValue> input_ops;
  util::Map<std::string /* produce/consume key */, Argument> input_output_args;
  for (const XrtEdge *edge : node->in_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument &arg = edge->argument();
      CHECK_GT(operands_.count(arg), 0);
      const TrtValue &operand = operands_.at(arg);
      input_ops.emplace(arg, operand);
      const std::string &k = arg.meta_data().consume_key;
      input_output_args.emplace(k, arg);
    }
  }
  for (const XrtEdge *edge : node->out_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument &arg = edge->argument();
      const std::string &k = arg.meta_data().produce_key;
      input_output_args.emplace(k, arg);
    }
  }

  size_t num_outputs = input_output_args.size() - input_ops.size();
  CHECK_GE(num_outputs, 0) << "Outputs number should >= 0.";
  context_param->op_name = node->name();
  context_param->builder = builder_.get();
  context_param->message = OpMessage(node);
  context_param->arguments = std::move(input_output_args);
  context_param->inputs = std::move(input_ops);
  context_param->num_outputs = num_outputs;
}

std::shared_ptr<Executable> TrtGraphCompiler::Compile(
    const XrtGraph *graph, const std::vector<Parameter> &entry_params,
    const std::vector<Parameter> &return_params, const std::vector<InputOutputAlias> &aliases) {
  // Build entry and return trt values.
  PopulateEntryParams(entry_params);
  // PopulateReturnParams(return_params);

  algorithm::TopologyVisit(*graph, [&](const XrtNode *node) {
    TrtOpContext::Param param;
    SetupKernelContextParam(node, &param);
    TrtOpContext op_context(param);
    // Do compile
    auto op_kernel = BuildOpKernel(node->type());
    op_kernel->Compile(&op_context);

    // Always insert the new output into `operands_`.
    const auto &outputs = op_context.outputs();
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
      operands_[it->first] = it->second;
    }
  });

  for (int i = 0; i < return_params.size(); ++i) {
    Argument arg = ArgFromParameter(return_params[i]);
    const TrtValue &value = operands_.at(arg);
    builder_->MarkOutput(value.handle());
  }

  // return std::make_shared<TrtExecutable>(builder_->BuildCudaEngine());
  return std::make_shared<TrtExecutable>(builder_->ReleaseBuilder(), builder_->ReleaseNetwork(),
                                         builder_->host_weights());
}

REGISTER_GRAPH_COMPILER(XrtEngine::TENSORRT, TrtGraphCompiler);

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
