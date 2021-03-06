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
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/autograd/autograd_meta.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {

namespace one {

TensorInfo::TensorInfo(const Tensor& tensor) : shape_(tensor.shape()), dtype_(tensor.dtype()) {}

Maybe<Tensor> TensorInfo::zeros() const { return functional::Constant(*shape_.get(), 0, dtype_); }

}  // namespace one

}  // namespace oneflow
