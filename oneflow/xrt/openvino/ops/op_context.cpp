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
#include "oneflow/xrt/openvino/ops/op_context.h"

namespace oneflow {
namespace xrt {
namespace openvino {

std::shared_ptr<ngraph::Node> OpenvinoOpContext::Input(const std::string& name) {
  return Input(ArgumentFromKey(name));
}

std::shared_ptr<ngraph::Node> OpenvinoOpContext::Output(const std::string& name) {
  return Output(ArgumentFromKey(name));
}

std::shared_ptr<ngraph::Node> OpenvinoOpContext::Input(const Argument& arg) {
  CHECK_GT(param_.inputs.count(arg), 0);
  return param_.inputs.at(arg);
}

std::shared_ptr<ngraph::Node> OpenvinoOpContext::Output(const Argument& arg) {
  CHECK_GT(outputs_.count(arg), 0);
  return outputs_.at(arg);
}

void OpenvinoOpContext::SetOutput(const std::string& name,
                                  const std::shared_ptr<ngraph::Node> ngraph_node) {
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
