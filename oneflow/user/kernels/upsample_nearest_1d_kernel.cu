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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/user/kernels/upsample_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpsampleNearest1DForward(const int64_t elem_cnt, const T* in_dptr,
                                         NdIndexOffsetHelper<int64_t, 3> in_helper,
                                         NdIndexOffsetHelper<int64_t, 3> out_helper,
                                         const int64_t in_height, const float scale_factor,
                                         T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h;
    out_helper.OffsetToNdIndex(index, n, c, h);
    const int64_t in_h = GetNearestInputIndex(h, scale_factor, in_height);
    out_dptr[index] = in_dptr[in_helper.NdIndexToOffset(n, c, in_h)];
  }
}

template<typename T>
__global__ void UpsampleNearest1DBackward(const int64_t elem_cnt, const T* dy_dptr,
                                          NdIndexOffsetHelper<int64_t, 3> dy_helper,
                                          NdIndexOffsetHelper<int64_t, 3> dx_helper,
                                          const int64_t in_height, const float scale_factor,
                                          T* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h;
    dy_helper.OffsetToNdIndex(index, n, c, h);
    const int64_t dx_h = GetNearestInputIndex(h, scale_factor, in_height);
    cuda::atomic::Add(dx_dptr + dx_helper.NdIndexToOffset(n, c, dx_h), dy_dptr[index]);
  }
}

}  // namespace

template<typename T>
class UpsampleNearest1DGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearest1DGPUKernel() = default;
  ~UpsampleNearest1DGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float height_scale = ctx->Attr<float>("scale_factor");
    const int64_t elem_cnt = y_blob->shape().elem_cnt();
    const int64_t in_height = x_blob->shape().At(2);
    const int64_t out_height = y_blob->shape().At(2);
    if (in_height == out_height) {
      Memcpy<DeviceType::kGPU>(ctx->device_ctx(), y_blob->mut_dptr<void>(), x_blob->dptr<void>(),
                               x_blob->shape().elem_cnt() * GetSizeOfDataType(x_blob->data_type()));
    } else {
      NdIndexOffsetHelper<int64_t, 3> in_helper(x_blob->shape().At(0), x_blob->shape().At(1),
                                                x_blob->shape().At(2));
      NdIndexOffsetHelper<int64_t, 3> out_helper(y_blob->shape().At(0), y_blob->shape().At(1),
                                                 y_blob->shape().At(2));
      RUN_CUDA_KERNEL((UpsampleNearest1DForward<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                      x_blob->dptr<T>(), in_helper, out_helper, x_blob->shape().At(2),
                      1.f / height_scale, y_blob->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleNearestGrad1DGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearestGrad1DGPUKernel() = default;
  ~UpsampleNearestGrad1DGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_blob == nullptr) { return; }
    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx_blob->mut_dptr<T>(), 0,
                             dx_blob->shape().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const float height_scale = ctx->Attr<float>("scale_factor");
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    const int64_t in_height = dx_blob->shape().At(2);
    const int64_t out_height = dy_blob->shape().At(2);
    if (in_height == out_height) {
      Memcpy<DeviceType::kGPU>(
          ctx->device_ctx(), dx_blob->mut_dptr<void>(), dy_blob->dptr<void>(),
          dy_blob->shape().elem_cnt() * GetSizeOfDataType(dy_blob->data_type()));
    } else {
      NdIndexOffsetHelper<int64_t, 3> dy_helper(dy_blob->shape().At(0), dy_blob->shape().At(1),
                                                dy_blob->shape().At(2));
      NdIndexOffsetHelper<int64_t, 3> dx_helper(dx_blob->shape().At(0), dx_blob->shape().At(1),
                                                dx_blob->shape().At(2));
      RUN_CUDA_KERNEL((UpsampleNearest1DBackward<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                      dy_blob->dptr<T>(), dy_helper, dx_helper, dx_blob->shape().At(2),
                      1.f / height_scale, dx_blob->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPNEAREST1D_GPU_KERNEL(dtype)                                     \
  REGISTER_USER_KERNEL("upsample_nearest_1d")                                          \
      .SetCreateFn<UpsampleNearest1DGPUKernel<dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("upsample_nearest_1d_grad")                                     \
      .SetCreateFn<UpsampleNearestGrad1DGPUKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_UPSAMPNEAREST1D_GPU_KERNEL(float)
REGISTER_UPSAMPNEAREST1D_GPU_KERNEL(double)

}  // namespace oneflow
