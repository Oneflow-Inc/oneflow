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
#include <ngraph/op/constant.hpp>

#include "oneflow/xrt/openvino/ops/op_context.h"
#include "oneflow/xrt/openvino/ngraph_shape.h"

namespace oneflow {
namespace xrt {
namespace openvino {

std::shared_ptr<ngraph::Node> OpenvinoOpContext::Input(const std::string& name) {
  return Input(ArgumentFromKey(name));
}

std::shared_ptr<ngraph::Node> OpenvinoOpContext::Output(const std::string& name) {
  return Output(ArgumentFromKey(name));
}

std::shared_ptr<ngraph::Node> OpenvinoOpContext::Weight(const std::string& name) {
  return Weight(ArgumentFromKey(name));
}

std::shared_ptr<ngraph::Node> OpenvinoOpContext::Input(const Argument& arg) {
  if (param_.inputs.count(arg) > 0) { return param_.inputs.at(arg); }
  auto it = entry_params_map_.find(arg);
  CHECK(it != entry_params_map_.end());
  NgraphShape shape(it->second.shape(), it->second.data_type());
  std::shared_ptr<ngraph::Node> input_node = std::make_shared<ngraph::op::Parameter>(
      shape.data_type(), ngraph::PartialShape(shape.shape()));
  graph_inputs_[arg] = input_node;
  return input_node;
}

std::shared_ptr<ngraph::Node> OpenvinoOpContext::Output(const Argument& arg) {
  CHECK_GT(outputs_.count(arg), 0);
  return outputs_.at(arg);
}

std::shared_ptr<ngraph::Node> OpenvinoOpContext::Weight(const Argument& arg) {
  if (param_.inputs.count(arg) > 0) { return param_.inputs.at(arg); }
  auto it = entry_params_map_.find(arg);
  CHECK(it != entry_params_map_.end());
  NgraphShape shape(it->second.shape(), it->second.data_type());
  std::shared_ptr<ngraph::Node> input_node =
      std::make_shared<ngraph::op::Constant>(shape.data_type(), shape.shape(), it->second.data());
  graph_weight_[arg] = input_node;
  return input_node;
}

void OpenvinoOpContext::SetOutput(const std::string& name,
                                  const std::shared_ptr<ngraph::Node>& ngraph_node) {
  Argument arg = ArgumentFromKey(name);
  outputs_[arg] = ngraph_node;
}

DataType OpenvinoOpContext::InputType(const std::string& name) const {
  return ArgumentFromKey(name).data_type();
}

DataType OpenvinoOpContext::OutputType(const std::string& name) const {
  return ArgumentFromKey(name).data_type();
}

Shape OpenvinoOpContext::InputShape(const std::string& name) const {
  return ArgumentFromKey(name).shape();
}

Shape OpenvinoOpContext::OutputShape(const std::string& name) const {
  return ArgumentFromKey(name).shape();
}

Argument OpenvinoOpContext::ArgumentFromKey(const std::string& key) const {
  CHECK_GT(param_.arguments.count(key), 0);
  return param_.arguments.at(key);
}

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow
