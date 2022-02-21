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

#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_METHODS_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_METHODS_H_

#include "oneflow/core/framework/tensor.h"

namespace oneflow {
namespace one {

class Tensor;

namespace view {

bool IsViewApplicable(const std::shared_ptr<Tensor>& input);

Maybe<Tensor> BasicView(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                        int64_t storage_offset);

Maybe<Tensor> BasicView(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                        const Stride& target_stride, int64_t storage_offset);

Maybe<Tensor> Reshape(const std::shared_ptr<Tensor>& input, const Shape& shape);

Maybe<Tensor> Slice(const std::shared_ptr<Tensor>& input, const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends, const std::vector<int64_t>& steps);

Maybe<Tensor> Unsqueeze(const std::shared_ptr<Tensor>& input, const int32_t& expand_dim);

Maybe<Tensor> Squeeze(const std::shared_ptr<Tensor>& input,
                      const std::vector<int32_t>& squeeze_dims);

Maybe<Tensor> Permute(const std::shared_ptr<Tensor>& input, const std::vector<int32_t>& permute);

}  // namespace view
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_METHOD_H_
