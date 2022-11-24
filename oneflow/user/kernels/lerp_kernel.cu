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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

template<typename T>
__global__ void LerpForwardGpu(const int n, const T* start, const T* end, const T* weight, T*out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = start[i] + weight[i] * (end[i] - start[i]); }
}

template<typename T>
__global__ void LerpBackwardGpu(const int n, const T* start, const T* end, const T* weight, const T*out_diff,
                                T* start_diff, T* end_diff, T* weight_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) { 
    start_diff[i] = (static_cast<T>(1.0) - weight[i]) * out_diff[i];
    end_diff[i] = weight[i] * out_diff[i];
    weight_diff[i] = (end[i] - start[i]) * out_diff[i];
  }
}

namespace {

template<typename T>
class CudaLerpKernel final : public user_op::OpKernel {
 public:
  CudaLerpKernel() = default;
  ~CudaLerpKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* start = ctx->Tensor4ArgNameAndIndex("start", 0);
    const user_op::Tensor* end = ctx->Tensor4ArgNameAndIndex("end", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t start_elem_cnt = start->shape_view().elem_cnt();
    const int64_t end_elem_cnt = end->shape_view().elem_cnt();
    const int64_t weight_elem_cnt = weight->shape_view().elem_cnt();
    CHECK_EQ(start_elem_cnt, end_elem_cnt);
    CHECK_EQ(start_elem_cnt, weight_elem_cnt);

    const T* start_ptr = start->dptr<T>();
    const T* end_ptr = end->dptr<T>();
    const T* weight_ptr = weight->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    RUN_CUDA_KERNEL((LerpForwardGpu<T>), ctx->stream(), start_elem_cnt, start_elem_cnt, start_ptr, end_ptr, weight_ptr, out_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_LERP_KERNEL(dtype)                                \
  REGISTER_USER_KERNEL("lerp")                                         \
      .SetCreateFn<CudaLerpKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_LERP_KERNEL(float)
REGISTER_GPU_LERP_KERNEL(double)
REGISTER_GPU_LERP_KERNEL(uint8_t)
REGISTER_GPU_LERP_KERNEL(int8_t)
REGISTER_GPU_LERP_KERNEL(int32_t)
REGISTER_GPU_LERP_KERNEL(int64_t)
REGISTER_GPU_LERP_KERNEL(half)

template<typename T>
class CudaLerpGradKernel final : public user_op::OpKernel {
 public:
  CudaLerpGradKernel() = default;
  ~CudaLerpGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* start = ctx->Tensor4ArgNameAndIndex("start", 0);
    const user_op::Tensor* end = ctx->Tensor4ArgNameAndIndex("end", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* out_diff = ctx->Tensor4ArgNameAndIndex("out_diff", 0);
    user_op::Tensor* start_diff = ctx->Tensor4ArgNameAndIndex("start_diff", 0);
    user_op::Tensor* end_diff = ctx->Tensor4ArgNameAndIndex("end_diff", 0);
    user_op::Tensor* weight_diff = ctx->Tensor4ArgNameAndIndex("weight_diff", 0);

    const int64_t start_elem_cnt = start->shape_view().elem_cnt();
    const int64_t end_elem_cnt = end->shape_view().elem_cnt();
    const int64_t weight_elem_cnt = weight->shape_view().elem_cnt();
    CHECK_EQ(start_elem_cnt, end_elem_cnt);
    CHECK_EQ(start_elem_cnt, weight_elem_cnt);

    const T* start_ptr = start->dptr<T>();
    const T* end_ptr = end->dptr<T>();
    const T* weight_ptr = weight->dptr<T>();
    const T* out_diff_ptr = out_diff->dptr<T>();
    T* start_diff_ptr = start_diff->mut_dptr<T>();
    T* end_diff_ptr = end_diff->mut_dptr<T>();
    T* weight_diff_ptr = weight_diff->mut_dptr<T>();

    RUN_CUDA_KERNEL((LerpBackwardGpu<T>), ctx->stream(), start_elem_cnt, 
                    start_elem_cnt, start_ptr, end_ptr, weight_ptr, out_diff_ptr, 
                    start_diff_ptr, end_diff_ptr, weight_diff_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_LERP_GRAD_KERNEL(dtype)                                \
  REGISTER_USER_KERNEL("lerp_grad")                                         \
      .SetCreateFn<CudaLerpGradKernel<dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)      \
                       && (user_op::HobDataType("start_diff", 0) == GetDataType<dtype>::value));

REGISTER_GPU_LERP_GRAD_KERNEL(float)
REGISTER_GPU_LERP_GRAD_KERNEL(double)
REGISTER_GPU_LERP_GRAD_KERNEL(uint8_t)
REGISTER_GPU_LERP_GRAD_KERNEL(int8_t)
REGISTER_GPU_LERP_GRAD_KERNEL(int32_t)
REGISTER_GPU_LERP_GRAD_KERNEL(int64_t)
REGISTER_GPU_LERP_GRAD_KERNEL(half)

}  // namespace

}  // namespace oneflow