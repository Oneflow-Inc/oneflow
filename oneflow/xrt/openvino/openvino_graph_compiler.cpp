#include "oneflow/xrt/openvino/openvino_graph_compiler.h"
#include "oneflow/xrt/node_util.h"
#include "oneflow/xrt/openvino/ops/op_kernel.h"
#include <ngraph/op/constant.hpp>

namespace oneflow {
namespace xrt {
namespace openvino {

void OpenvinoGraphCompiler::PopulateEntryParams(const std::vector<Parameter> &entry_params,
                                                ngraph::ParameterVector *inputs_nodes,
                                                util::Map<std::string, int> &in_out_to_param_idx) {
  for (int i = 0; i < entry_params.size(); ++i) {
    Argument arg = ArgFromParameter(entry_params[i]);
    NgraphShape shape(entry_params[i].shape(), entry_params[i].data_type());
    std::shared_ptr<ngraph::Node> input_node;
    if (entry_params[i].is_model()) {
      input_node = std::make_shared<ngraph::op::Constant>(shape.data_type(), shape.shape(),
                                                          entry_params[i].data());
    } else {
      input_node = std::make_shared<ngraph::op::Parameter>(shape.data_type(),
                                                           ngraph::PartialShape(shape.shape()));
      in_out_to_param_idx[input_node->get_friendly_name()] = i;
      inputs_nodes->push_back(ngraph::as_type_ptr<ngraph::op::Parameter>(input_node));
    }
    operands_[arg] = input_node;
  }
}

Argument OpenvinoGraphCompiler::ArgFromParameter(const Parameter &param) {
  return Argument(param.name(), param.shape(), param.data_type());
}

void OpenvinoGraphCompiler::SetupKernelContextParam(const XrtNode *node,
                                                    OpenvinoOpContext::Param *context_param) {
  util::Map<Argument, std::shared_ptr<ngraph::Node>> input_ops;
  util::Map<std::string /* produce/consume key */, Argument> input_output_args;
  for (const XrtEdge *edge : node->in_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument &arg = edge->argument();
      CHECK_GT(operands_.count(arg), 0);
      std::shared_ptr<ngraph::Node> ngraph_node = operands_.at(arg);
      input_ops.emplace(arg, ngraph_node);
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
  context_param->op_name = node->name();
  context_param->message = OpMessage(node);
  context_param->arguments = std::move(input_output_args);
  context_param->inputs = std::move(input_ops);
}

std::shared_ptr<Executable> OpenvinoGraphCompiler::Compile(
    const XrtGraph *graph, const std::vector<Parameter> &entry_params,
    const std::vector<Parameter> &return_params, const std::vector<InputOutputAlias> &aliases) {
  // openvino input output name to entry and return param index.
  util::Map<std::string, int> in_out_to_param_idx;
  ngraph::ParameterVector parameter_nodes;
  PopulateEntryParams(entry_params, &parameter_nodes, in_out_to_param_idx);

  algorithm::TopologyVisit(*graph, [&](const XrtNode *node) {
    OpenvinoOpContext::Param param;
    SetupKernelContextParam(node, &param);
    OpenvinoOpContext op_context(param);
    // Do compile
    auto op_kernel = BuildOpKernel(node->type());
    op_kernel->Compile(&op_context);

    // Always insert the new output into `operands_`.
    const auto &outputs = op_context.outputs();
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
      operands_[it->first] = it->second;
    }
  });

  ngraph::ResultVector result_nodes;
  for (int i = 0; i < return_params.size(); ++i) {
    Argument arg = ArgFromParameter(return_params[i]);
    std::shared_ptr<ngraph::Node> ngraph_node = operands_.at(arg);
    in_out_to_param_idx[ngraph_node->get_friendly_name()] = i;
    auto result = std::make_shared<ngraph::op::Result>(ngraph_node);
    result_nodes.push_back(result);
  }

  std::shared_ptr<ngraph::Function> ngraph_func =
      std::make_shared<ngraph::Function>(result_nodes, parameter_nodes);
  InferenceEngine::CNNNetwork cnn_network(ngraph_func);
  InferenceEngine::Core ie;
  InferenceEngine::InputsDataMap input_info(cnn_network.getInputsInfo());
  for (auto &input : input_info) {
    auto it = in_out_to_param_idx.find(input.first);
    CHECK(it != in_out_to_param_idx.end());
    const int input_idx = it->second;
    InferenceEngineDataDesc data_desc(entry_params[input_idx].shape(),
                                      entry_params[input_idx].data_type());
    input.second->setPrecision(data_desc.precision());
    input.second->setLayout(data_desc.layout());
  }
  auto executable_network =
      std::make_unique<InferenceEngine::ExecutableNetwork>(ie.LoadNetwork(cnn_network, "CPU"));
  return std::make_shared<OpenvinoExecutable>(std::move(executable_network), in_out_to_param_idx);
}

REGISTER_GRAPH_COMPILER(XrtEngine::OPENVINO, OpenvinoGraphCompiler);

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
