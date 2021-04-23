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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {

TensorArg::TensorArg() : add2_op_(op_expr_helper::AddNOp(2).GetPtrOrThrow()) {}

bool TensorArg::Empty() const { return !acc_tensor_; }

void TensorArg::Release() {
  acc_tensor_.reset();
}

Maybe<void> TensorArg::PushPartialTensor(const std::shared_ptr<Tensor>& partial_tensor) {
  if (!acc_tensor_) {
    acc_tensor_ = partial_tensor;
  } else {
    acc_tensor_ = JUST(OpInterpUtil::Dispatch<Tensor>(*add2_op_, {partial_tensor, acc_tensor_}));
  }
  return Maybe<void>::Ok();
}

Maybe<Tensor> TensorArg::GetAccTensor() {
  CHECK_OR_RETURN(Empty() == false) << "Can not GetAccTensor because it is empty";
  return acc_tensor_;
}

}  // namespace one
}  // namespace oneflow
