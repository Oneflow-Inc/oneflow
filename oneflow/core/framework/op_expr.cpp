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
#include "oneflow/core/framework/op_expr.h"

namespace oneflow {
namespace one {

UserOpExpr::UserOpExpr(const std::string& op_name, UserOpConf&& proto)
    : BuiltinOpExpr(op_name), proto_(proto) {
  for (const auto& it : proto_.input()) {
    for (const auto& input : it.second.s()) { indexed_input_names_.push_back(input); }
  }
  for (const auto& it : proto_.output()) {
    for (const auto& output : it.second.s()) { indexed_output_names_.push_back(output); }
  }
}

std::shared_ptr<OpExpr> UserOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new UserOpExpr);
}

VariableOpExpr::VariableOpExpr(const std::string& op_name, VariableOpConf&& proto)
    : BuiltinOpExpr(op_name), proto_(proto) {
  indexed_output_names_.push_back(proto_.out());
}

std::shared_ptr<OpExpr> VariableOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new VariableOpExpr);
}

CastToMirroredOpExpr::CastToMirroredOpExpr(const std::string& op_name, CastToMirroredOpConf&& proto)
    : BuiltinOpExpr(op_name), proto_(proto) {
  indexed_input_names_.push_back(proto_.in());
  indexed_output_names_.push_back(proto_.out());
}

std::shared_ptr<OpExpr> CastToMirroredOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new CastToMirroredOpExpr);
}

CastFromMirroredOpExpr::CastFromMirroredOpExpr(const std::string& op_name,
                                               CastFromMirroredOpConf&& proto)
    : BuiltinOpExpr(op_name), proto_(proto) {
  indexed_input_names_.push_back(proto_.in());
  indexed_output_names_.push_back(proto_.out());
}

std::shared_ptr<OpExpr> CastFromMirroredOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new CastFromMirroredOpExpr);
}

DistributeSplitOpExpr::DistributeSplitOpExpr(const std::string& op_name,
                                             DistributeSplitOpConf&& proto)
    : BuiltinOpExpr(op_name), proto_(proto) {
  indexed_input_names_.push_back(proto_.in());
  for (const std::string& output : proto_.out()) { indexed_output_names_.push_back(output); }
}

std::shared_ptr<OpExpr> DistributeSplitOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new DistributeSplitOpExpr);
}

DistributeCloneOpExpr::DistributeCloneOpExpr(const std::string& op_name,
                                             DistributeCloneOpConf&& proto)
    : BuiltinOpExpr(op_name), proto_(proto) {
  indexed_input_names_.push_back(proto_.in());
  for (const std::string& output : proto_.out()) { indexed_output_names_.push_back(output); }
}

std::shared_ptr<OpExpr> DistributeCloneOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new DistributeCloneOpExpr);
}

DistributeConcatOpExpr::DistributeConcatOpExpr(const std::string& op_name,
                                               DistributeConcatOpConf&& proto)
    : BuiltinOpExpr(op_name), proto_(proto) {
  for (const std::string& input : proto_.in()) { indexed_input_names_.push_back(input); }
  indexed_output_names_.push_back(proto_.out());
}

std::shared_ptr<OpExpr> DistributeConcatOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new DistributeConcatOpExpr);
}

DistributeAddOpExpr::DistributeAddOpExpr(const std::string& op_name, DistributeAddOpConf&& proto)
    : BuiltinOpExpr(op_name), proto_(proto) {
  for (const std::string& input : proto_.in()) { indexed_input_names_.push_back(input); }
  indexed_output_names_.push_back(proto_.out());
}

std::shared_ptr<OpExpr> DistributeAddOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new DistributeAddOpExpr);
}

std::shared_ptr<OpExpr> FunctionOpExpr::GetBackwardOpExpr() const {
  // TODO(hjchen2)
  return std::shared_ptr<OpExpr>(new FunctionOpExpr);
}

}  // namespace one
}  // namespace oneflow
