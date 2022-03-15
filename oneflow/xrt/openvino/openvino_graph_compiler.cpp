/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/xrt/openvino/openvino_graph_compiler.h"
#include "oneflow/xrt/node_util.h"
#include "oneflow/xrt/openvino/ops/op_kernel.h"

namespace oneflow {
namespace xrt {
namespace openvino {

void OpenvinoGraphCompiler::PopulateEntryParams(const std::vector<Parameter>& entry_params,
                                                util::Map<Argument, Parameter>& entry_params_map,
                                                util::Map<Argument, int>& entry_params_index_map) {
  for (int i = 0; i < entry_params.size(); ++i) {
    Argument arg = ArgFromParameter(entry_params[i]);
    entry_params_map[arg] = entry_params[i];
    entry_params_index_map[arg] = i;
  }
}

Argument OpenvinoGraphCompiler::ArgFromParameter(const Parameter& param) {
  return Argument(param.name(), param.shape(), param.data_type());
}

void OpenvinoGraphCompiler::SetupKernelContextParam(const XrtNode* node,
                                                    OpenvinoOpContext::Param* context_param) {
  util::Map<Argument, std::shared_ptr<ngraph::Node>> input_ops;
  util::Map<std::string /* produce/consume key */, Argument> input_output_args;
  int input_size = 0;
  for (const XrtEdge* edge : node->in_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument& arg = edge->argument();
      const std::string& k = arg.meta_data().consume_key;
      input_size++;
      input_output_args.emplace(k, arg);
      // When arg is graph input, operands_ not hold it.
      if (operands_.count(arg) <= 0) { continue; }
      const auto& ngraph_node = operands_.at(arg);
      input_ops.emplace(arg, ngraph_node);
    }
  }
  for (const XrtEdge* edge : node->out_edges()) {
    if (!edge->IsControlEdge()) {
      const Argument& arg = edge->argument();
      const std::string& k = arg.meta_data().produce_key;
      input_output_args.emplace(k, arg);
    }
  }
  context_param->op_name = node->name();
  context_param->message = OpMessage(node);
  context_param->arguments = std::move(input_output_args);
  context_param->inputs = std::move(input_ops);
  context_param->input_size = input_size;
}

std::shared_ptr<Executable> OpenvinoGraphCompiler::Compile(
    const XrtGraph* graph, const std::vector<Parameter>& entry_params,
    const std::vector<Parameter>& return_params, const std::vector<InputOutputAlias>& aliases) {
  // openvino input output name to entry and return param index.
  util::Map<std::string, int> in_out_to_param_idx;
  ngraph::ParameterVector parameter_nodes;
  util::Map<Argument, Parameter> entry_params_map;
  util::Map<Argument, int> entry_params_index_map;
  PopulateEntryParams(entry_params, entry_params_map, entry_params_index_map);

  algorithm::TopologyVisit(*graph, [&](const XrtNode* node) {
    OpenvinoOpContext::Param param;
    SetupKernelContextParam(node, &param);
    OpenvinoOpContext op_context(param, entry_params_map);
    // Do compile
    auto op_kernel = BuildOpKernel(node->type());
    op_kernel->Compile(&op_context);

    const auto& graph_inputs = op_context.graph_inputs();
    for (auto it = graph_inputs.begin(); it != graph_inputs.end(); ++it) {
      operands_[it->first] = it->second;
      in_out_to_param_idx[it->second->get_friendly_name()] = entry_params_index_map[it->first];
      parameter_nodes.push_back(ngraph::as_type_ptr<ngraph::op::Parameter>(it->second));
    }
    const auto& graph_weight = op_context.graph_weight();
    for (auto it = graph_weight.begin(); it != graph_weight.end(); ++it) {
      operands_[it->first] = it->second;
    }
    // Always insert the new output into `operands_`.
    const auto& outputs = op_context.outputs();
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
  for (auto& input : input_info) {
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
