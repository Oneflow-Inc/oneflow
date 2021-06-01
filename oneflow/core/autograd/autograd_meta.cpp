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

#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/autograd/autograd_meta.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {

namespace one {

TensorInfo::TensorInfo(const Tensor& tensor) : shape_(tensor.shape()), dtype_(tensor.dtype()) {}

Maybe<Tensor> TensorInfo::zeros() const {
  const auto& interpreter = JUST(OpInterpUtil::GetInterpreter());
  const auto& zeros_op = JUST(op_expr_helper::ZerosOp(*shape_.get(), dtype_->data_type()));
  TensorTuple outputs(1);
  JUST(interpreter->Apply(*zeros_op, {}, &outputs));
  return outputs.at(0);
}

}  // namespace one

}  // namespace oneflow
