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
#include "oneflow/xrt/tvm/ops/op_context.h"

namespace oneflow {
namespace xrt {
namespace of_tvm {

TVMOpContext::TVMOpContext(const XrtNode* node, const PbMessage* message,
                           util::Map<Argument, tvm::relay::Expr>&& input_arg2expr,
                           util::Vector<Argument>&& output_args)
    : OpContext(*message), node_(node), input_name2expr_(), input_name2arg_(), output_name2arg_(), output_name2expr_() {
  for (const auto& pair : input_arg2expr) {
    std::string input_name = pair.first.meta_data().consume_key;
    input_name2expr_.emplace(input_name, pair.second);
    input_name2arg_.emplace(input_name, pair.first);
  }
  for (const auto arg: output_args) {
    std::string output_name = arg.meta_data().produce_key;
    output_name2arg_.emplace(output_name, arg);
  }
}

tvm::relay::Expr TVMOpContext::GetExpr4InputName(const std::string& name) const {
  auto it = input_name2expr_.find(name);
  CHECK(it != input_name2expr_.end())
      << "Cannot find input_name: " << name << " in TVMOpContext of node: " << node_->name();
  return it->second;
}

const Shape& TVMOpContext::GetShape4InputName(const std::string& name) const {
  auto it = input_name2arg_.find(name);
  CHECK(it != input_name2arg_.end())
      << "Cannot find input_name: " << name << " in TVMOpContext of node: " << node_->name();
  return it->second.shape();
}

const Shape& TVMOpContext::GetShape4OutputName(const std::string& name) const {
  auto it = output_name2arg_.find(name);
  CHECK(it != output_name2arg_.end())
      << "Cannot find output_name: " << name << " in TVMOpContext of node: " << node_->name();
  return it->second.shape();
}

tvm::relay::Expr TVMOpContext::GetExpr4OutputName(const std::string& name) const {
  auto it = output_name2expr_.find(name);
  CHECK(it != output_name2expr_.end())
      << "Cannot find output_name: " << name << " in TVMOpContext of node: " << node_->name();
  return it->second;
}

void TVMOpContext::SetExpr4OutputName(const std::string& name, tvm::relay::Expr&& expr) {
  CHECK(output_name2expr_.emplace(name, std::move(expr)).second);
}

std::string TVMOpContext::DebugStr() {
  std::string s;
  s += "\nin_exprs: ";
  for(const auto& pair : input_name2expr_) {
    s += pair.first;
    s += ",";
  }
  s += "\n input_arg: ";
  for(const auto& pair : input_name2arg_) {
    s += pair.first;
    s += ",";
  }
  s += "\n out_arg: ";
  for(const auto& pair : output_name2arg_) {
    s += pair.first;
    s += ",";
  }
  s += "\n output_expr: ";
  for(const auto& pair : output_name2expr_) {
    s += pair.first;
    s += ",";
  }
  s += "\n";
  return s;
} 

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow