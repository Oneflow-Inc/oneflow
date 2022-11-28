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
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {
namespace one {

bool TensorArg::Empty() const { return !acc_tensor_; }

void TensorArg::Release() { acc_tensor_.reset(); }

Maybe<void> TensorArg::PushPartialTensor(const std::shared_ptr<Tensor>& partial_tensor) {
  if (!acc_tensor_) {
    acc_tensor_ = partial_tensor;
  } else {
    // Should not inplace accumulate grad. For example,
    // >>> z = x + y
    // >>> p = x / z
    // >>> p.sum().backward()
    //
    // As we know that dx = dz + dp / z and dy = dz, so it will lead to wrong value
    // for dy if dx is shared with dz.
    acc_tensor_ =
        JUST(functional::Add(partial_tensor, acc_tensor_, /*alpha=*/1, /*inplace=*/false));
  }
  return Maybe<void>::Ok();
}

Maybe<Tensor> TensorArg::GetAccTensor() const {
  CHECK_OR_RETURN(Empty() == false) << "Can not GetAccTensor because it is empty";
  return acc_tensor_;
}

}  // namespace one
}  // namespace oneflow
