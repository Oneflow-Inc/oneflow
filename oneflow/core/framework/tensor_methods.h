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

class Stream;

namespace one {

class Tensor;

namespace view {

bool IsEnvViewDisabled();

bool IsViewApplicable(const std::shared_ptr<Tensor>& input);

static bool IsOverlappingMemorys(const std::vector<int64_t>& sizes,
                                 const std::vector<int64_t>& strides);

static int64_t MinStorageSize(const std::vector<int64_t>& sizes,
                              const std::vector<int64_t>& strides, int64_t storage_offset);

Maybe<Tensor> BasicView(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                        const int64_t storage_offset);

Maybe<Tensor> BasicView(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                        const Stride& target_stride, const int64_t storage_offset);

Maybe<void> InplaceView(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                        const Stride& target_stride, int64_t const storage_offset);

Maybe<Tensor> Reshape(const std::shared_ptr<Tensor>& input, const Shape& target_shape);

Maybe<Tensor> Reshape(const std::shared_ptr<Tensor>& input, const Shape& target_shape,
                      const Stride& target_stride);

Maybe<Tensor> Slice(const std::shared_ptr<Tensor>& input, const std::vector<int64_t>& starts,
                    const std::vector<int64_t>& ends, const std::vector<int64_t>& steps);

Maybe<Tensor> Unsqueeze(const std::shared_ptr<Tensor>& input, const int32_t expand_dim);

Maybe<void> InplaceUnsqueeze(const std::shared_ptr<Tensor>& input, const int32_t expand_dim);

Maybe<Tensor> Squeeze(const std::shared_ptr<Tensor>& input,
                      const std::vector<int32_t>& squeeze_dims);

Maybe<void> InplaceSqueeze(const std::shared_ptr<Tensor>& input,
                           const std::vector<int32_t>& squeeze_dims);

Maybe<Tensor> Expand(const std::shared_ptr<Tensor>& input, const Shape& expand_shape);

Maybe<void> InplaceExpand(const std::shared_ptr<Tensor>& input, const Shape& expand_shape);

Maybe<Tensor> Narrow(const std::shared_ptr<Tensor>& input, const int64_t dim, const int64_t start,
                     const int64_t length);

Maybe<Tensor> AsStridedGrad(const std::shared_ptr<one::Tensor>& dy,
                            const std::shared_ptr<one::Tensor>& input,
                            const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides,
                            const int64_t storage_offset);

Maybe<Tensor> AsStrided(const std::shared_ptr<one::Tensor>& input,
                        const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides,
                        const int64_t storage_offset);

Maybe<void> InplaceAsStrided(const std::shared_ptr<one::Tensor>& input,
                             const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides,
                             const int64_t storage_offset);

Maybe<Tensor> Transpose(const std::shared_ptr<Tensor>& input, const std::vector<int32_t>& permute);

Maybe<Tensor> UnfoldTensor(const std::shared_ptr<Tensor>& input, const int32_t dimension,
                           const int32_t size, const int32_t step);

Maybe<Tensor> Diagonal(const std::shared_ptr<Tensor>& input, const int32_t offset,
                       const int32_t dim1, const int32_t dim2);

}  // namespace view

Maybe<void> Touch(std::shared_ptr<Tensor> input, Symbol<Stream> stream);

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_METHOD_H_
