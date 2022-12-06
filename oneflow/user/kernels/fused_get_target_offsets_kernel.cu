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
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/common/primitive/binary_functor.h"

namespace oneflow {

namespace {

template<typename T>
struct ModFunctor {
  __device__ T Compute(T input_tensor) const {
    return fmod(input_tensor, static_cast<T>(1.0));
  }
};

template<>
struct ModFunctor<half> {
  ModFunctor<float> float_functor;
  __device__ half Compute(half input_tensor) const {
    return __float2half(float_functor.Compute(__half2float(input_tensor)));
  }
};

template<typename T>
struct GetStatsFunctor {
  __device__ bool Compute(T input_tensor, T input_tensor_mod_1, float g) const {
    return (input_tensor_mod_1 < static_cast<T>(g)) && (input_tensor > static_cast<T>(1.0));
  }
};

template<typename MOD_FUNCTOR, typename GET_STATUS_FUNCTOR, typename T>
__global__ void FusedGetTargetOffsetsForward(MOD_FUNCTOR mod_functor, 
                                             GET_STATUS_FUNCTOR get_stats_functor, 
                                             const int n, const T* gxy, const T* gxi, 
                                             const float g, bool* output_tensor, 
                                             const int64_t rows) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T gxy_i = gxy[i];
    const T gxi_i = gxi[i];
    const T gxy_mod_1 = mod_functor.Compute(gxy_i);
    const T gxi_mod_1 = mod_functor.Compute(gxi_i);
    const bool stats_1 = get_stats_functor.Compute(gxy_i, gxy_mod_1, g);
    const bool stats_2 = get_stats_functor.Compute(gxi_i, gxi_mod_1, g);
    if (i % 2 == 0) { 
      const int64_t extra_cols = i / 2; 
      output_tensor[i - extra_cols + rows] = stats_1;
      output_tensor[i + n - extra_cols + rows] = stats_2;
    } else {
      const int64_t extra_cols = (i + n - 1) / 2;
      output_tensor[extra_cols + rows] = stats_1;
      output_tensor[n + extra_cols + rows] = stats_2;
    }
  }
}

}  // namespace

template<typename T>
class FusedGetTargetOffsetsKernel final : public user_op::OpKernel {
 public:
  FusedGetTargetOffsetsKernel() = default;
  ~FusedGetTargetOffsetsKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* gxy = ctx->Tensor4ArgNameAndIndex("gxy", 0);
    const user_op::Tensor* gxi = ctx->Tensor4ArgNameAndIndex("gxi", 0);
    const float g = ctx->Attr<float>("g");

    user_op::Tensor* j = ctx->Tensor4ArgNameAndIndex("j", 0);

    const int64_t elem_cnt = gxy->shape_view().elem_cnt();
    const int64_t rows = gxy->shape_view().At(0);

    ModFunctor<T> mod_functor{};
    GetStatsFunctor<T> get_stats_functor{};

    RUN_CUDA_KERNEL((FusedGetTargetOffsetsForward<decltype(mod_functor), decltype(get_stats_functor), T>), ctx->stream(), elem_cnt, 
                    mod_functor, get_stats_functor, elem_cnt,
                    gxy->dptr<T>(), gxi->dptr<T>(), g, j->mut_dptr<bool>(), rows);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_TARGET_OFFSETS_CUDA_KERNEL(dtype)                \
  REGISTER_USER_KERNEL("fused_yolov5_get_target_offsets")                          \
      .SetCreateFn<FusedGetTargetOffsetsKernel<dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)      \
                       && (user_op::HobDataType("j", 0) == DataType::kBool) \
                       && (user_op::HobDataType("gxy", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_TARGET_OFFSETS_CUDA_KERNEL(float)
REGISTER_FUSED_GET_TARGET_OFFSETS_CUDA_KERNEL(double)
REGISTER_FUSED_GET_TARGET_OFFSETS_CUDA_KERNEL(half)

}  // namespace oneflow
