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
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__device__ void upsample_increment_value_bounded_cuda(T* data, int64_t width, int64_t height,
                                                      int64_t element, int64_t x, int64_t y,
                                                      T value) {
  int64_t access_x = max(min(x, width - 1), static_cast<int64_t>(0));
  int64_t access_y = max(min(y, height - 1), static_cast<int64_t>(0));
  cuda::atomic::FastAdd(data, access_y * width + access_x, element, value);
}

template<typename T>
__global__ void UpsampleBicubic2dForward(const int64_t elem_cnt, const T* in_dptr,
                                         const int64_t nbatch, const int64_t channels,
                                         const int64_t in_height, const int64_t in_width,
                                         const int64_t out_height, const int64_t out_width,
                                         const float scale_height, const float scale_width,
                                         bool align_corners, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(idx, elem_cnt) {
    const int output_x = idx % out_width;
    const int output_y = idx / out_width;

    const T* in = in_dptr;
    T* out = out_dptr;

    const T real_x = GetAreaPixel(scale_width, output_x, align_corners, /*cubic=*/true);
    int64_t input_x = floor(1.0 * real_x);
    const T t_x = real_x - input_x;

    const T real_y = GetAreaPixel(scale_height, output_y, align_corners, /*cubic=*/true);
    int64_t input_y = floor(1.0 * real_y);
    const T t_y = real_y - input_y;

    for (int64_t c = 0; c < channels * nbatch; c++) {
      T coefficients[4];

      // Interpolate 4 times in the x direction
      for (int64_t i = 0; i < 4; i++) {
        coefficients[i] = cubic_interp1d<T>(
            upsample_get_value_bounded<T>(in, in_width, in_height, input_x - 1, input_y - 1 + i),
            upsample_get_value_bounded<T>(in, in_width, in_height, input_x + 0, input_y - 1 + i),
            upsample_get_value_bounded<T>(in, in_width, in_height, input_x + 1, input_y - 1 + i),
            upsample_get_value_bounded<T>(in, in_width, in_height, input_x + 2, input_y - 1 + i),
            t_x);
      }

      // Interpolate in the y direction using x interpolations
      out[output_y * out_width + output_x] = cubic_interp1d<T>(
          coefficients[0], coefficients[1], coefficients[2], coefficients[3], t_y);

      // Move to next channel
      in += in_width * in_height;
      out += out_width * out_height;
    }
  }
}

template<typename T>
__global__ void UpsampleBicubic2dBackward(const int64_t elem_cnt, const T* dy_dptr,
                                          const int64_t nbatch, const int64_t channels,
                                          const int64_t in_height, const int64_t in_width,
                                          const int64_t out_height, const int64_t out_width,
                                          const float scale_height, const float scale_width,
                                          bool align_corners, T* dx_dptr) {
  CUDA_1D_KERNEL_LOOP(idx, elem_cnt) {
    const int output_x = idx % out_width;
    const int output_y = idx / out_width;

    T* in = dx_dptr;
    const T* out = dy_dptr;

    T real_x = GetAreaPixel(scale_width, output_x, align_corners, true);
    int64_t input_x = floor(1.0 * real_x);
    T t_x = real_x - input_x;

    T real_y = GetAreaPixel(scale_height, output_y, align_corners, true);
    int64_t input_y = floor(1.0 * real_y);
    T t_y = real_y - input_y;

    T x_coeffs[4];
    T y_coeffs[4];

    get_cubic_upsample_coefficients<T>(x_coeffs, t_x);
    get_cubic_upsample_coefficients<T>(y_coeffs, t_y);

    for (int64_t c = 0; c < channels * nbatch; c++) {
      T out_value = out[output_y * out_width + output_x];

      for (int64_t i = 0; i < 4; i++) {
        for (int64_t j = 0; j < 4; j++) {
          upsample_increment_value_bounded_cuda<T>(in, in_width, in_height, elem_cnt,
                                                   input_x - 1 + i, input_y - 1 + j,
                                                   out_value * y_coeffs[j] * x_coeffs[i]);
        }
      }

      in += in_width * in_height;
      out += out_width * out_height;
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
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T* in_ptr = x_tensor->dptr<T>();
    T* out_ptr = y_tensor->mut_dptr<T>();
    const bool align_corners = ctx->Attr<bool>("align_corners");

    const int nbatch = x_tensor->shape_view().At(0);
    const int channels = x_tensor->shape_view().At(1);
    const int64_t in_height = x_tensor->shape_view().At(2);
    const int64_t in_width = x_tensor->shape_view().At(3);
    const int64_t out_height = y_tensor->shape_view().At(2);
    const int64_t out_width = y_tensor->shape_view().At(3);
    const std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    double height_scale = ctx->Attr<double>("height_scale");
    double width_scale = ctx->Attr<double>("width_scale");
    if (!output_size.empty()) {
      height_scale = static_cast<double>(out_height) / static_cast<double>(in_height);
      width_scale = static_cast<double>(out_width) / static_cast<double>(in_width);
    }
    const int64_t elem_cnt = out_height * out_width;

    if (in_height == out_height && in_width == out_width) {
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), y_tensor->mut_dptr<void>(), x_tensor->dptr<void>(),
          x_tensor->shape_view().elem_cnt() * GetSizeOfDataType(x_tensor->data_type()));
    } else {
      const T scale_height = GetAreaPixelScale(in_height, out_height, align_corners, height_scale);
      const T scale_width = GetAreaPixelScale(in_width, out_width, align_corners, width_scale);

      RUN_CUDA_KERNEL((UpsampleBicubic2dForward<T>), ctx->stream(), elem_cnt, elem_cnt,
                      x_tensor->dptr<T>(), nbatch, channels, in_height, in_width, out_height,
                      out_width, scale_height, scale_width, align_corners, y_tensor->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleBicubic2dGradGPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBicubic2dGradGPUKernel() = default;
  ~UpsampleBicubic2dGradGPUKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    Memset<DeviceType::kCUDA>(ctx->stream(), dx_tensor->mut_dptr<T>(), 0,
                              dx_tensor->shape_view().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const bool align_corners = ctx->Attr<bool>("align_corners");

    const int nbatch = dx_tensor->shape_view().At(0);
    const int channels = dx_tensor->shape_view().At(1);
    const int64_t in_height = dx_tensor->shape_view().At(2);
    const int64_t in_width = dx_tensor->shape_view().At(3);
    const int64_t out_height = dy_tensor->shape_view().At(2);
    const int64_t out_width = dy_tensor->shape_view().At(3);
    const std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
    double height_scale = ctx->Attr<double>("height_scale");
    double width_scale = ctx->Attr<double>("width_scale");
    if (!output_size.empty()) {
      height_scale = static_cast<double>(out_height) / static_cast<double>(in_height);
      width_scale = static_cast<double>(out_width) / static_cast<double>(in_width);
    }
    const int64_t elem_cnt = out_height * out_width;

    if (in_height == out_height && in_width == out_width) {
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), dx_tensor->mut_dptr<void>(), dy_tensor->dptr<void>(),
          dy_tensor->shape_view().elem_cnt() * GetSizeOfDataType(dy_tensor->data_type()));
    } else {
      const T scale_height = GetAreaPixelScale(in_height, out_height, align_corners, height_scale);
      const T scale_width = GetAreaPixelScale(in_width, out_width, align_corners, width_scale);

      RUN_CUDA_KERNEL((UpsampleBicubic2dBackward<T>), ctx->stream(), elem_cnt, elem_cnt,
                      dy_tensor->dptr<T>(), nbatch, channels, in_height, in_width, out_height,
                      out_width, scale_height, scale_width, align_corners,
                      dx_tensor->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_BICUBIC_CUDA_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("upsample_bicubic_2d")                                           \
      .SetCreateFn<UpsampleBicubic2dGPUKernel<dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("upsample_bicubic_2d_grad")                                      \
      .SetCreateFn<UpsampleBicubic2dGradGPUKernel<dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_UPSAMPLE_BICUBIC_CUDA_KERNEL(float)
REGISTER_UPSAMPLE_BICUBIC_CUDA_KERNEL(double)

}  // namespace oneflow
