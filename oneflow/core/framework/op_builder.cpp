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

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/op_builder.h"

namespace oneflow {
namespace one {

/*static*/ TensorNameScope* TensorNameScope::Global() {
  static TensorNameScope scope;
  return &scope;
}

const std::string& TensorNameScope::Lookup(const std::shared_ptr<Tensor>& tensor) const {
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t key = reinterpret_cast<uint64_t>(tensor.get());
  const auto& it = tensor_names_.find(key);
  if (it != tensor_names_.end()) {
    return it->second;
  } else {
    return default_tensor_name_;
  }
}

void TensorNameScope::Record(const std::shared_ptr<Tensor>& tensor, const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t key = reinterpret_cast<uint64_t>(tensor.get());
  // We assume that the name of the tensor will be update more than once.
  tensor_names_[key] = name;
}

OpBuilder::OpBuilder(const std::string& op_type_name) {
  *(proto_.mutable_op_type_name()) = op_type_name;
}

OpBuilder& OpBuilder::Op(const std::string& op_type_name) {
  *(proto_.mutable_op_type_name()) = op_type_name;
  return *this;
}

OpBuilder& OpBuilder::Input(const std::string& input_name) { return this->Input(input_name, 1); }

OpBuilder& OpBuilder::Input(const std::string& input_name, const int count) {
  CHECK_GT(count, 0);
  CHECK_EQ(proto_.input().count(input_name), 0)
      << "The Input " << input_name << " has been specified more than once.";
  auto& input_list = (*(proto_.mutable_input()))[input_name];
  for (int i = 0; i < count; ++i) {
    const std::string& tensor_name = "^#Position_" + std::to_string(input_pos_++);
    input_list.mutable_s()->Add()->assign(tensor_name);
    indexed_input_names_.push_back(tensor_name);
  }
  return *this;
}

OpBuilder& OpBuilder::Output(const std::string& output_name) {
  return this->Output(output_name, 1);
}

OpBuilder& OpBuilder::Output(const std::string& output_name, const int count) {
  CHECK_GT(count, 0);
  CHECK_EQ(proto_.output().count(output_name), 0)
      << "The output " << output_name << " has been specified more than once.";
  auto& output_list = (*(proto_.mutable_output()))[output_name];
  for (int i = 0; i < count; ++i) {
    const std::string& tensor_name = op_name_ + "/" + output_name + "_" + std::to_string(i);
    output_list.mutable_s()->Add()->assign(tensor_name);
    indexed_output_names_.push_back(tensor_name);
  }
  return *this;
}

OpBuilder& OpBuilder::Attr(const std::string& attr_name, const AttrValue& attr_value) {
  (*(proto_.mutable_attr()))[attr_name] = attr_value;
  return *this;
}

OpBuilder& OpBuilder::Attr(const std::string& attr_name, const std::string& serialized_attr_value) {
  AttrValue attr_value;
  TxtString2PbMessage(serialized_attr_value, &attr_value);
  (*(proto_.mutable_attr()))[attr_name] = attr_value;
  return *this;
}

std::shared_ptr<UserOpExpr> OpBuilder::Build() {
  return std::make_shared<UserOpExpr>(op_name_, std::move(proto_), indexed_input_names_,
                                      indexed_output_names_);
}

}  // namespace one
}  // namespace oneflow
