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
#include <glog/logging.h>

#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {
namespace one {

/*static*/ TensorNameScope* TensorNameScope::Global() {
  static TensorNameScope scope;
  return &scope;
}

const std::string& TensorNameScope::Lookup(const TensorRef& tensor) const {
  std::lock_guard<std::mutex> lock(mutex_);

  uint64_t key = reinterpret_cast<uint64_t>(tensor.get());
  const auto& it = tensor_names_.find(key);
  if (it != tensor_names_.end()) {
    return it->second;
  } else {
    return default_tensor_name_;
  }
}

void TensorNameScope::Record(const TensorRef& tensor, const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);

  uint64_t key = reinterpret_cast<uint64_t>(tensor.get());
  // We assume that the same tensor will only be recorded once.
  CHECK_EQ(tensor_names_.count(key), 0);
  tensor_names_.emplace(key, name);
}

OpBuilder::OpBuilder(const std::string& op_type_name) : operation_(new Operation) {
  *(operation_->proto.mutable_op_type_name()) = op_type_name;
}

OpBuilder& OpBuilder::Op(const std::string& op_type_name) {
  *(operation_->proto.mutable_op_type_name()) = op_type_name;
  return *this;
}

OpBuilder& OpBuilder::Input(const std::string& input_name, const std::vector<TensorRef>& input) {
  CHECK_GT(input.size(), 0);
  auto& input_list = (*(operation_->proto.mutable_input()))[input_name];
  for (const auto& tensor : input) {
    input_list.mutable_s()->Add()->assign(TensorNameScope::Global()->Lookup(tensor));
  }
  return *this;
}

OpBuilder& OpBuilder::Output(const std::string& output_name) {
  return this->Output(output_name, 1);
}

OpBuilder& OpBuilder::Output(const std::string& output_name, const int count) {
  CHECK_GT(count, 0);
  auto& output_list = (*(operation_->proto.mutable_output()))[output_name];
  const std::string& op_name = operation_->op_name;
  for (int i = 0; i < count; ++i) {
    output_list.mutable_s()->Add()->assign(op_name + "/" + output_name + "_" + std::to_string(i));
  }
  return *this;
}

OpBuilder& OpBuilder::Attr(const std::string& attr_name, const AttrValue& attr_value) {
  (*(operation_->proto.mutable_attr()))[attr_name] = attr_value;
  return *this;
}

OpBuilder& OpBuilder::Attr(const std::string& attr_name, const std::string& serialized_attr_value) {
  AttrValue attr_value;
  TxtString2PbMessage(serialized_attr_value, &attr_value);
  (*(operation_->proto.mutable_attr()))[attr_name] = attr_value;
  return *this;
}

std::shared_ptr<Operation>&& OpBuilder::Build() {
  for (const auto& it : operation_->proto.input()) {
    for (const auto& input : it.second.s()) { operation_->indexed_input_names.push_back(input); }
  }
  return std::move(operation_);
}

}  // namespace one
}  // namespace oneflow
