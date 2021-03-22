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

#include "oneflow/core/framework/tensor_arg.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/tensor_tuple.h"

namespace oneflow {
namespace one {

namespace {

Maybe<UserOpExpr> GetAddNOpExpr(int32_t n) {
  return OpBuilder("add_n").Input("in", n).Output("out").Build();
}

}  // namespace

bool TensorArg::Empty() const { return partial_sum_tensors_.empty() && acc_tensor_; }

void TensorArg::Release() {
  partial_sum_tensors_.clear();
  acc_tensor_.reset();
}

void TensorArg::PushPartialTensor(const std::shared_ptr<Tensor>& partial_tensor) {
  partial_sum_tensors_.push_back(partial_tensor);
}

Maybe<Tensor> TensorArg::GetAccTensor() {
  if (!acc_tensor_) {
    size_t input_num = partial_sum_tensors_.size();
    TensorTuple input(input_num);
    TensorTuple output(1);
    for (size_t i = 0; i < input_num; ++i) { input.at(i) = partial_sum_tensors_.at(i); }
    const auto& add_n = CHECK_JUST(GetAddNOpExpr(input_num));
    GetInterpreter()->Apply(add_n, input, &output);
    acc_tensor_ = output.at(0);
    partial_sum_tensors_.clear();
  }
  return acc_tensor_;
}

}  // namespace one
}  // namespace oneflow
