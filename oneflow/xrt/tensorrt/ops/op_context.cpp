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
#include "oneflow/xrt/tensorrt/ops/op_context.h"
#include "oneflow/xrt/tensorrt/trt_value.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

const std::string& TrtOpContext::SoleOutputName() const {
  CHECK_EQ(num_outputs(), 1);
  return param_.output_names.front();
}

nvinfer1::ITensor* TrtOpContext::Input(const std::string& name) {
  return Input(ArgumentFromKey(name));
}

nvinfer1::ITensor* TrtOpContext::Output(const std::string& name) {
  return Output(ArgumentFromKey(name));
}

nvinfer1::ITensor* TrtOpContext::Input(const Argument& arg) {
  CHECK_GT(param_.inputs.count(arg), 0);
  return param_.inputs.at(arg).AsTensor(builder());
}

nvinfer1::ITensor* TrtOpContext::Output(const Argument& arg) {
  CHECK_GT(outputs_.count(arg), 0);
  return outputs_.at(arg).AsTensor(builder());
}

nvinfer1::ITensor* TrtOpContext::SoleInput() {
  CHECK_EQ(num_inputs(), 1);
  auto it = param_.inputs.begin();
  return (it->second).AsTensor(builder());
}

nvinfer1::ITensor* TrtOpContext::SoleOutput() {
  CHECK_EQ(outputs_.size(), 1);
  auto it = outputs_.begin();
  return (it->second).AsTensor(builder());
}

nvinfer1::Weights& TrtOpContext::Weight(const std::string& name) {
  return Weight(ArgumentFromKey(name));
}

nvinfer1::Weights& TrtOpContext::Weight(const Argument& arg) {
  CHECK_GT(param_.inputs.count(arg), 0);
  return param_.inputs.at(arg).AsWeight(builder());
}

void TrtOpContext::SetOutput(const std::string& name, nvinfer1::ITensor* tensor) {
  SetOutput(name, TrtValue::Tensor(builder(), tensor));
}

void TrtOpContext::SetOutput(const std::string& name, const TrtValue& value) {
  Argument arg = ArgumentFromKey(name);
  outputs_[arg] = value;
  nvinfer1::ITensor* tensor = builder()->GetTensor(value.handle());
  tensor->setName(arg.name().c_str());
}

void TrtOpContext::SetSoleOutput(nvinfer1::ITensor* tensor) {
  CHECK_EQ(outputs_.size(), 0);
  SetOutput(SoleOutputName(), tensor);
}

DataType TrtOpContext::InputType(const std::string& name) const {
  return ArgumentFromKey(name).data_type();
}

DataType TrtOpContext::SoleInputType() const {
  CHECK_EQ(num_inputs(), 1);
  auto it = param_.inputs.begin();
  return (it->first).data_type();
}

DataType TrtOpContext::OutputType(const std::string& name) const {
  return ArgumentFromKey(name).data_type();
}

DataType TrtOpContext::SoleOutputType() const {
  return ArgumentFromKey(SoleOutputName()).data_type();
}

Shape TrtOpContext::InputShape(const std::string& name) const {
  return ArgumentFromKey(name).shape();
}

Shape TrtOpContext::SoleInputShape() const {
  CHECK_EQ(num_inputs(), 1);
  auto it = param_.inputs.begin();
  return (it->first).shape();
}

Shape TrtOpContext::OutputShape(const std::string& name) const {
  return ArgumentFromKey(name).shape();
}

Shape TrtOpContext::SoleOutputShape() const { return ArgumentFromKey(SoleOutputName()).shape(); }

bool TrtOpContext::HasInput(const std::string& name) const {
  return param_.arguments.count(name) > 0;
}

Argument TrtOpContext::ArgumentFromKey(const std::string& key) const {
  CHECK_GT(param_.arguments.count(key), 0);
  return param_.arguments.at(key);
}

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
