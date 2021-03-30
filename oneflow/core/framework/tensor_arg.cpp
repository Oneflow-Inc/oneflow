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
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/op_interpreter_util.h"

namespace oneflow {
namespace one {

bool TensorArg::Empty() const { return partial_sum_tensors_.empty() && !acc_tensor_; }

void TensorArg::Release() {
  partial_sum_tensors_.clear();
  acc_tensor_.reset();
}

void TensorArg::PushPartialTensor(const std::shared_ptr<Tensor>& partial_tensor) {
  partial_sum_tensors_.push_back(partial_tensor);
}

Maybe<Tensor> TensorArg::GetAccTensor() {
  CHECK_OR_RETURN(Empty() == false) << "Can not GetAccTensor because it is empty";
  if (!acc_tensor_) {
    size_t input_num = partial_sum_tensors_.size();
    if (input_num == 1) {
      acc_tensor_ = partial_sum_tensors_.at(0);
    } else {
      TensorTuple input(input_num);
      for (size_t i = 0; i < input_num; ++i) { input.at(i) = partial_sum_tensors_.at(i); }
      TensorTuple output(1);
      const auto& add_n = JUST(op_expr_helper::AddNOp(input_num));
      JUST(JUST(OpInterpUtil::GetInterpreter())->Apply(*add_n, input, &output));
      acc_tensor_ = output.at(0);
    }
    partial_sum_tensors_.clear();
  }
  return acc_tensor_;
}

}  // namespace one
}  // namespace oneflow
