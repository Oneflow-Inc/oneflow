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
#ifndef ONEFLOW_USER_KERNELS_LOSS_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_LOSS_KERNEL_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace user_op {
namespace loss {

#ifdef WITH_CUDA
template<typename T>
struct Float16To;

template<>
struct Float16To<const float16*> {
  using type = const half*;
};

template<>
struct Float16To<float16*> {
  using type = half*;
};

#define FLOAT16_TO_HALF(x) \
  Float16To<decltype(x)>::type x##_ = reinterpret_cast<Float16To<decltype(x)>::type>(x);

#endif  // WITH_CUDA

enum class ReductionType { kNone, kSum, kMean, kNotImplemented };

inline ReductionType GetReductionType(const std::string& reduction) {
  if (reduction == "mean") {
    return ReductionType::kMean;
  } else if (reduction == "none") {
    return ReductionType::kNone;
  } else if (reduction == "sum") {
    return ReductionType::kSum;
  }
  UNIMPLEMENTED();
  return ReductionType::kNotImplemented;
}

template<typename T>
void ApplyLossReductionIfNeed(int64_t elem_cnt, const T* tmp_out, T* out,
                              const ReductionType reduction_type);

template<typename T>
void ApplyLossReductionIfNeed(DeviceCtx* ctx, int64_t elem_cnt, const T* tmp_out, T* out,
                              const ReductionType reduction_type);

template<typename T>
user_op::InferTmpSizeFn GenDefaultInferTmpSizeFn(const std::string& input_name = "input",
                                                 const std::string& reduction_name = "reduction") {
  return [=](user_op::InferContext* ctx) {
    const int64_t n = ctx->InputShape(input_name, 0).elem_cnt();
    const ReductionType reduction = GetReductionType(ctx->Attr<std::string>(reduction_name));

    if (reduction != ReductionType::kNone) { return GetCudaAlignedSize(n * sizeof(T)); }
    return static_cast<size_t>(0);
  };
}

}  // namespace loss
}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_LOSS_KERNEL_UTIL_H_
