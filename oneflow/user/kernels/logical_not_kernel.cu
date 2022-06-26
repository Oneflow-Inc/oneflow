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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
struct LogicalNotFunctor {
  OF_DEVICE_FUNC bool operator()(T x) const { return !x; }
};

}  // namespace

template<typename T, typename K>
class GpuLogicalNotKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  GpuLogicalNotKernel() = default;
  ~GpuLogicalNotKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int64_t elem_cnt = x->shape_view().elem_cnt();
    OF_CUDA_CHECK(
        (cuda::elementwise::Unary(LogicalNotFunctor<T>(), elem_cnt, y->mut_dptr<K>(), x->dptr<T>(),
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_LOGICAL_NOT_KERNEL(dtype, DataType)              \
  REGISTER_USER_KERNEL("logical_not")                                  \
      .SetCreateFn<GpuLogicalNotKernel<dtype, bool>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == DataType));

OF_PP_FOR_EACH_TUPLE(REGISTER_CUDA_LOGICAL_NOT_KERNEL, ARITHMETIC_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ);
OF_PP_FOR_EACH_TUPLE(REGISTER_CUDA_LOGICAL_NOT_KERNEL, HALF_DATA_TYPE_SEQ);

}  // namespace oneflow
