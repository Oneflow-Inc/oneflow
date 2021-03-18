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
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/op_builder.h"

namespace oneflow {
namespace one {

static constexpr char _PositionalPlaceholderPrefix[] = "_/#^Placeholder_";

OpBuilder::OpBuilder(const std::string& op_type_name) {
  *(proto_.mutable_op_type_name()) = op_type_name;
  op_name_ = *CHECK_JUST(UniqueStr(op_type_name));
}

OpBuilder::OpBuilder(const std::string& op_type_name, const std::string& op_name)
    : op_name_(op_name) {
  *(proto_.mutable_op_type_name()) = op_type_name;
}

Maybe<OpBuilder&> OpBuilder::MaybeInput(const std::string& input_name, const int count) {
  CHECK_GT_OR_RETURN(count, 0);
  CHECK_EQ_OR_RETURN(proto_.input().count(input_name), 0)
      << "The Input " << input_name << " has been specified more than once.";
  auto& input_list = (*(proto_.mutable_input()))[input_name];
  for (int i = 0; i < count; ++i) {
    const std::string& tensor_name = _PositionalPlaceholderPrefix + std::to_string(input_pos_++);
    input_list.mutable_s()->Add()->assign(tensor_name);
    indexed_ibns_.push_back(input_name + "_" + std::to_string(i));
  }
  return *this;
}

OpBuilder& OpBuilder::Input(const std::string& input_name) {
  return CHECK_JUST(MaybeInput(input_name, 1));
}
OpBuilder& OpBuilder::Input(const std::string& input_name, const int count) {
  return CHECK_JUST(MaybeInput(input_name, count));
}

Maybe<OpBuilder&> OpBuilder::MaybeOutput(const std::string& output_name, const int count) {
  CHECK_GT_OR_RETURN(count, 0);
  CHECK_EQ_OR_RETURN(proto_.output().count(output_name), 0)
      << "The output " << output_name << " has been specified more than once.";
  auto& output_list = (*(proto_.mutable_output()))[output_name];
  for (int i = 0; i < count; ++i) {
    const std::string& tensor_name = op_name_ + "/" + output_name + "_" + std::to_string(i);
    output_list.mutable_s()->Add()->assign(tensor_name);
    indexed_obns_.push_back(output_name + "_" + std::to_string(i));
  }
  return *this;
}

OpBuilder& OpBuilder::Output(const std::string& output_name) {
  return CHECK_JUST(MaybeOutput(output_name, 1));
}

OpBuilder& OpBuilder::Output(const std::string& output_name, const int count) {
  return CHECK_JUST(MaybeOutput(output_name, count));
}

Maybe<OpBuilder&> OpBuilder::MaybeAttr(const std::string& attr_name,
                                       const cfg::AttrValue& attr_value) {
  AttrValue pb_attr_value;
  attr_value.ToProto(&pb_attr_value);
  (*(proto_.mutable_attr()))[attr_name] = pb_attr_value;
  return *this;
}

OpBuilder& OpBuilder::Attr(const std::string& attr_name, const cfg::AttrValue& attr_value) {
  return CHECK_JUST(MaybeAttr(attr_name, attr_value));
}

Maybe<UserOpExpr> OpBuilder::Build() {
  return std::make_shared<UserOpExpr>(op_name_, std::move(proto_), indexed_ibns_, indexed_obns_);
}

}  // namespace one
}  // namespace oneflow
