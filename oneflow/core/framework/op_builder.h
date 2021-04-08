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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_BUILDER_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_BUILDER_H_

#include <string>

#include "oneflow/core/framework/op_expr.h"

namespace oneflow {
namespace one {

// The op builder for UserOp.
// Note that the internal proto will be moved if the Build method is called.
// Therefore, please make sure that the Build method be called at last, and do not perform any
// operations on this builder instance after the calling.
class OpBuilder {
 public:
  OpBuilder() = delete;
  explicit OpBuilder(const std::string& op_type_name);
  explicit OpBuilder(const std::string& op_type_name, const std::string& op_name);
  virtual ~OpBuilder() = default;

  Maybe<OpBuilder&> MaybeInput(const std::string& input_name, const int count);
  OpBuilder& Input(const std::string& input_name);
  OpBuilder& Input(const std::string& input_name, const int count);

  Maybe<OpBuilder&> MaybeOutput(const std::string& output_name, const int count);
  OpBuilder& Output(const std::string& output_name);
  OpBuilder& Output(const std::string& output_name, const int count);

  Maybe<OpBuilder&> MaybeAttr(const std::string& attr_name, const AttrValue& attr_value);
  OpBuilder& Attr(const std::string& attr_name, const AttrValue& attr_value);

  // TODO(): Set attribute from primitive type.
  // template <typename T>
  // OpBuilder& Attr(const std::string& attr_name, const T& attr_value);

  Maybe<UserOpExpr> Build();

 private:
  std::string op_name_;
  UserOpConf proto_;

  int input_pos_ = 0;
  std::vector<std::string> indexed_ibns_;
  std::vector<std::string> indexed_obns_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_BUILDER_H_
