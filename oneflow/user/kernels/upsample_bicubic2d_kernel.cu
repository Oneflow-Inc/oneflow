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
__global__ void UpsampleBicubic2dForward(const int64_t elem_cnt, const T* in_dptr,
                                         NdIndexOffsetHelper<int64_t, 4> in_helper,
                                         NdIndexOffsetHelper<int64_t, 4> out_helper,
                                         const int64_t in_height, const int64_t in_width,
                                         const float scale_h, const float scale_w, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    out_helper.OffsetToNdIndex(index, n, c, h, w);

    T real_x = scale_w * w;
    int64_t input_x = real_x;
    T t_x = real_x - input_x;

    T real_y = scale_h * h;
    int64_t input_y = real_y;
    T t_y = real_y - input_y;

    T coefficients[4];

    // Interpolate 4 times in the x direction
    for (int64_t i = 0; i < 4; i++) {
      coefficients[i] = cubic_interp1d<T>(
          upsample_get_value_bounded<T>(in_dptr, in_width, in_height, input_x - 1, input_y - 1 + i),
          upsample_get_value_bounded<T>(in_dptr, in_width, in_height, input_x + 0, input_y - 1 + i),
          upsample_get_value_bounded<T>(in_dptr, in_width, in_height, input_x + 1, input_y - 1 + i),
          upsample_get_value_bounded<T>(in_dptr, in_width, in_height, input_x + 2, input_y - 1 + i),
          t_x);
    }
    out_dptr[index] =
        cubic_interp1d<T>(coefficients[0], coefficients[1], coefficients[2], coefficients[3], t_y);
  }
}

template<typename T>
__global__ void UpsampleBicubic2dBackward(const int64_t elem_cnt, const T* dy_dptr,
                                          NdIndexOffsetHelper<int64_t, 4> dy_helper,
                                          NdIndexOffsetHelper<int64_t, 4> dx_helper,
                                          const int64_t dx_height, const int64_t dx_width,
                                          const float scale_h, const float scale_w, T* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t n, c, h, w;
    dy_helper.OffsetToNdIndex(index, n, c, h, w);

    T real_x = scale_w * w;
    int64_t input_x = real_x;
    T t_x = real_x - input_x;

    T real_y = scale_h * h;
    int64_t input_y = real_y;
    T t_y = real_y - input_y;

    T x_coeffs[4], y_coeffs[4];

    get_cubic_upsample_coefficients<T>(x_coeffs, t_x);
    get_cubic_upsample_coefficients<T>(y_coeffs, t_y);

    for (int64_t i = 0; i < 4; i++) {
      for (int64_t j = 0; j < 4; j++) {
        cuda::atomic::Add(
            dx_dptr + dx_helper.NdIndexToOffset(n, c, input_y - 1 + j, input_x - 1 + i),
            dy_dptr[index]);
      }
    }
  }
}

}  // namespace

template<typename T>
class UpsampleBicubic2dGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBicubic2dGPUKernel() = default;
  ~UpsampleBicubic2dGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const bool align_corners = ctx->Attr<float>("align_corners");
    const int64_t elem_cnt = y_blob->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> in_helper(x_blob->shape().At(0), x_blob->shape().At(1),
                                              x_blob->shape().At(2), x_blob->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> out_helper(y_blob->shape().At(0), y_blob->shape().At(1),
                                               y_blob->shape().At(2), y_blob->shape().At(3));
    const int64_t in_height = x_blob->shape().At(2);
    const int64_t in_width = x_blob->shape().At(3);
    const int64_t out_height = y_blob->shape().At(2);
    const int64_t out_width = y_blob->shape().At(3);
    const T scale_height = GetAreaPixelScale(in_height, out_height, align_corners, height_scale);
    const T scale_width = GetAreaPixelScale(in_width, out_width, align_corners, width_scale);

    RUN_CUDA_KERNEL((UpsampleBicubic2dForward<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    x_blob->dptr<T>(), in_helper, out_helper, x_blob->shape().At(2),
                    x_blob->shape().At(3), scale_height, scale_width, y_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleBicubic2dGradGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBicubic2dGradGPUKernel() = default;
  ~UpsampleBicubic2dGradGPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_blob == nullptr) { return; }
    Memset<DeviceType::kGPU>(ctx->device_ctx(), dx_blob->mut_dptr<T>(), 0,
                             dx_blob->shape().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const bool align_corners = ctx->Attr<float>("align_corners");
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    NdIndexOffsetHelper<int64_t, 4> dy_helper(dy_blob->shape().At(0), dy_blob->shape().At(1),
                                              dy_blob->shape().At(2), dy_blob->shape().At(3));
    NdIndexOffsetHelper<int64_t, 4> dx_helper(dx_blob->shape().At(0), dx_blob->shape().At(1),
                                              dx_blob->shape().At(2), dx_blob->shape().At(3));
    const int64_t in_height = dx_blob->shape().At(2);
    const int64_t in_width = dx_blob->shape().At(3);
    const int64_t out_height = dy_blob->shape().At(2);
    const int64_t out_width = dy_blob->shape().At(3);
    const T scale_height = GetAreaPixelScale(in_height, out_height, align_corners, height_scale);
    const T scale_width = GetAreaPixelScale(in_width, out_width, align_corners, width_scale);

    RUN_CUDA_KERNEL((UpsampleBicubic2dBackward<T>), ctx->device_ctx(), elem_cnt, elem_cnt,
                    dy_blob->dptr<T>(), dy_helper, dx_helper, dx_blob->shape().At(2),
                    dx_blob->shape().At(3), scale_height, scale_width, dx_blob->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_BICUBIC_GPU_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("upsample_bicubic_2d")                                          \
      .SetCreateFn<UpsampleBicubic2dGPUKernel<dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("upsample_bicubic_2d_grad")                                     \
      .SetCreateFn<UpsampleBicubic2dGradGPUKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_UPSAMPLE_BICUBIC_GPU_KERNEL(float)
REGISTER_UPSAMPLE_BICUBIC_GPU_KERNEL(double)

}  // namespace oneflow
