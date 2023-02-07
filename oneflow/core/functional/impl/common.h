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
#ifndef ONEFLOW_CORE_FUNCTIONAL_IMPL_COMMON_H_
#define ONEFLOW_CORE_FUNCTIONAL_IMPL_COMMON_H_

#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/common/maybe.h"
#include "fmt/core.h"

namespace oneflow {
namespace one {
namespace functional {

static constexpr size_t kMaxInputCount = 128;
static constexpr size_t kMaxOutputCount = 128;

bool IsStaticZerosTensor(const std::shared_ptr<Tensor>& x);
bool IsInplaceValid(const std::shared_ptr<Tensor>& x);
bool IsScalarTensor(const std::shared_ptr<Tensor>& x);

Maybe<std::vector<int32_t>> CheckAxis(const std::vector<int32_t>& axis, const int32_t& ndim);
Maybe<void> CheckInplaceValid(const std::shared_ptr<Tensor>& x);
Maybe<void> CheckInplaceCastValid(const std::shared_ptr<Tensor>& x,
                                  const std::shared_ptr<Tensor>& x_cast);
Maybe<void> CheckInplaceShapeCanExpandTo(const Shape& shape, const Shape& expand_shape);
Optional<Stride> ComputeStride(const Shape& shape, const Stride& stride, const Shape& target_shape);
Maybe<Shape> InferShapeUnspecifiedDim(const int64_t& elem_count, const Shape& shape);

// returns unified_shape
Maybe<Shape> InferUnifiedShapeForBroadcasting(const std::vector<Shape>& shapes);
// returns tuple<unified_shape, need_to_broadcasts>
Maybe<std::tuple<Shape, std::deque<bool>>> InferUnifiedShapeForBroadcastingWithInfo(
    const std::vector<Shape>& shapes);

Maybe<void> BroadcastSeedToAllRanks(uint64_t* seed, int64_t root = 0);

Maybe<std::vector<int32_t>> GetPermWhenTransposeAxisToLastDim(const int32_t& ndim,
                                                              const int32_t& axis);
Maybe<std::vector<int32_t>> GetInversedPerm(const std::vector<int32_t>& perm);

// batchify function is referenced from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Convolution.cpp#L729
Maybe<std::tuple<std::shared_ptr<Tensor>, bool>> batchify(const std::shared_ptr<Tensor>& input,
                                                          const int64_t num_spatial_dims,
                                                          const std::string& func_name);
}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_IMPL_COMMON_H_
