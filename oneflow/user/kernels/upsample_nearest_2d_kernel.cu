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
#include "oneflow/user/kernels/upsample_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpsampleNearest2DForward(const int64_t elem_cnt, const T* in_dptr,
                                         NdIndexOffsetHelper<int64_t, 4> in_helper,
                                         NdIndexOffsetHelper<int64_t, 4> out_helper,
                                         const int64_t in_height, const int64_t in_width,
                                         const float scale_h, const float scale_w, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    out_helper.OffsetToNdIndex(index, n, c, h, w);
    const int64_t in_h = GetNearestInputIndex(h, scale_h, in_height);
    const int64_t in_w = GetNearestInputIndex(w, scale_w, in_width);
    out_dptr[index] = in_dptr[in_helper.NdIndexToOffset(n, c, in_h, in_w)];
  }
}

template<typename T>
__global__ void UpsampleNearest2DBackward(const int64_t elem_cnt, const T* dy_dptr,
                                          NdIndexOffsetHelper<int64_t, 4> dy_helper,
                                          NdIndexOffsetHelper<int64_t, 4> dx_helper,
                                          const int64_t dx_height, const int64_t dx_width,
                                          const float scale_h, const float scale_w, T* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    dy_helper.OffsetToNdIndex(index, n, c, h, w);
    const int64_t dx_h = GetNearestInputIndex(h, scale_h, dx_height);
    const int64_t dx_w = GetNearestInputIndex(w, scale_w, dx_width);
    *(dx_dptr + dx_helper.NdIndexToOffset(n, c, dx_h, dx_w)) += dy_dptr[index];
  }
}

}  // namespace

template<typename T>
class UpsampleNearest2DGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearest2DGPUKernel() = default;
  ~UpsampleNearest2DGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const int64_t elem_cnt = y_blob->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> in_helper(x_blob->shape().At(0), x_blob->shape().At(1),
                                              x_blob->shape().At(2), x_blob->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> out_helper(y_blob->shape().At(0), y_blob->shape().At(1),
                                               y_blob->shape().At(2), y_blob->shape().At(3));
    RUN_CUDA_KERNEL((UpsampleNearest2DForward<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    x_blob->dptr<T>(), in_helper, out_helper, x_blob->shape().At(2),
                    x_blob->shape().At(3), 1.f / height_scale, 1.f / width_scale,
                    y_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleNearest2DGradGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleNearest2DGradGPUKernel() = default;
  ~UpsampleNearest2DGradGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_blob == nullptr) { return; }
    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx_blob->mut_dptr<T>(), 0,
                             dx_blob->shape().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> dy_helper(dy_blob->shape().At(0), dy_blob->shape().At(1),
                                              dy_blob->shape().At(2), dy_blob->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> dx_helper(dx_blob->shape().At(0), dx_blob->shape().At(1),
                                              dx_blob->shape().At(2), dx_blob->shape().At(3));
    RUN_CUDA_KERNEL((UpsampleNearest2DBackward<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    dy_blob->dptr<T>(), dy_helper, dx_helper, dx_blob->shape().At(2),
                    dx_blob->shape().At(3), 1.f / height_scale, 1.f / width_scale,
                    dx_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_NEAREST_GPU_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("upsample_nearest_2d")                                          \
      .SetCreateFn<UpsampleNearest2DGPUKernel<dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("upsample_nearest_2d_grad")                                     \
      .SetCreateFn<UpsampleNearest2DGradGPUKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_UPSAMPLE_NEAREST_GPU_KERNEL(float)
REGISTER_UPSAMPLE_NEAREST_GPU_KERNEL(double)

}  // namespace oneflow
