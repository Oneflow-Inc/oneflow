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


template<typename T>
class UpsampleBicubic2dCPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBicubic2dCPUKernel() = default;
  ~UpsampleBicubic2dCPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    T* in_ptr = x_blob->dptr<T>();
    T* out_ptr = y_blob->mut_dptr<T>();
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const bool align_corners = ctx->Attr<bool>("align_corners");

    const int nbatch = x_blob->shape().At(0);
    const int channels = x_blob->shape().At(1);
    const int64_t in_height = x_blob->shape().At(2);
    const int64_t in_width = x_blob->shape().At(3);
    const int64_t out_height = y_blob->shape().At(2);
    const int64_t out_width = y_blob->shape().At(3);

    if(in_height == out_height && in_width == out_width){
      for (int64_t output_y = 0; output_y < out_height; output_y++) {
        for (int64_t output_x = 0; output_x < out_width; output_x++) {
          scalar_t* in = &in_ptr[output_y * in_width + output_x];
          scalar_t* out = &out_ptr[output_y * out_width + output_x];
          for (int64_t c = 0; c < channels; ++c) {
            in[0] = out[0];
            in += in_width * in_height;
            out += out_width * out_height;
          }
        }
      }
      return;
    }

    const T scale_height = GetAreaPixelScale(in_height, out_height, align_corners, height_scale);
    const T scale_width = GetAreaPixelScale(in_width, out_width, align_corners, width_scale);

    
    for (int64_t output_y = 0; output_y < out_height; output_y++) {
      for(int64_t output_x = 0; output_x < out_width; output_x++) {
        T *in = in_ptr;
        T *out = out_ptr;

        const T real_x = scale_width * output_x;
        int64_t input_x = real_x;
        const T t_x = real_x - input_x;

        const T real_y = scale_height * output_y;
        int64_t input_y = real_y;
        const T t_y = real_y - input_y;

         for (int64_t c = 0; c < channels * nbatch; c++) {
            T coefficients[4];

            // Interpolate 4 times in the x direction
            for (int64_t i = 0; i < 4; i++) {
              coefficients[i] = cubic_interp1d<T>(
                  upsample_get_value_bounded<T>(
                      in, in_width, in_height, input_x - 1, input_y - 1 + i),
                  upsample_get_value_bounded<T>(
                      in, in_width, in_height, input_x + 0, input_y - 1 + i),
                  upsample_get_value_bounded<T>(
                      in, in_width, in_height, input_x + 1, input_y - 1 + i),
                  upsample_get_value_bounded<T>(
                      in, in_width, in_height, input_x + 2, input_y - 1 + i),
                  t_x);
            }

            // Interpolate in the y direction using x interpolations
            out[output_y * out_width + output_x] = cubic_interp1d<T>(
                coefficients[0],
                coefficients[1],
                coefficients[2],
                coefficients[3],
                t_y);

            // Move to next channel
            in += in_width * in_height;
            out += out_width * out_height;
          }
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class UpsampleBicubic2dGradCPUKernel final : public user_op::OpKernel {
 public:
  UpsampleBicubic2dGradCPUKernel() = default;
  ~UpsampleBicubic2dGradCPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    if (dx_blob == nullptr) { return; }
    Memset<DeviceType::kCPU>(ctx->device_ctx(), dx_blob->mut_dptr<T>(), 0,
                             dx_blob->shape().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    T* in_ptr = dx_blob->dptr<T>();
    T* out_ptr = dy_blob->mut_dptr<T>();
    const float height_scale = ctx->Attr<float>("height_scale");
    const float width_scale = ctx->Attr<float>("width_scale");
    const bool align_corners = ctx->Attr<bool>("align_corners");
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();

    const int nbatch = dx_blob->shape().At(0);
    const int channels = dx_blob->shape().At(1);
    channels = channels * nbatch;
    const int64_t in_height = dx_blob->shape().At(2);
    const int64_t in_width = dx_blob->shape().At(3);
    const int64_t out_height = dy_blob->shape().At(2);
    const int64_t out_width = dy_blob->shape().At(3);

    if(in_height == out_height && in_width == out_width){
      for (int64_t output_y = 0; output_y < out_height; output_y++) {
        for (int64_t output_x = 0; output_x < out_width; output_x++) {
          T* in = &in_ptr[output_y * in_width + output_x];
          T* out = &out_ptr[output_y * out_width + output_x];
          for (int64_t c = 0; c < channels; ++c) {
            in[0] = out[0];
            in += in_width * in_height;
            out += out_width * out_height;
          }
        }
      }
      return;
    }

    const T scale_height = GetAreaPixelScale(in_height, out_height, align_corners, height_scale);
    const T scale_width = GetAreaPixelScale(in_width, out_width, align_corners, width_scale);

    for (int64_t output_y = 0; output_y < out_height; output_y++) {
      for (int64_t output_x = 0; output_x < out_width; output_x++) {
        T* in = in_ptr;
        T* out = out_ptr;

        T real_x = scale_width * output_x;
        int64_t input_x = real_x;
        T t_x = real_x - input_x;

        T real_y = scale_height * output_y;
        int64_t input_y = real_y;
        T t_y = real_y - input_y;

        T x_coeffs[4];
        T y_coeffs[4];

        get_cubic_upsample_coefficients<T>(x_coeffs, t_x);
        get_cubic_upsample_coefficients<T>(y_coeffs, t_y);

        for (int64_t c = 0; c < channels; c++) {
          T out_value = out[output_y * output_width + output_x];

          for (int64_t i = 0; i < 4; i++) {
            for (int64_t j = 0; j < 4; j++) {
              upsample_increment_value_bounded<T>(
                  in,
                  input_width,
                  input_height,
                  input_x - 1 + i,
                  input_y - 1 + j,
                  out_value * y_coeffs[j] * x_coeffs[i]);
            }
          }

          in += input_width * input_height;
          out += output_width * output_height;
        }
      }
    }
    
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UPSAMPLE_BICUBIC_CPU_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("upsample_bicubic_2d")                                          \
      .SetCreateFn<UpsampleBicubic2dCPUKernel<dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("upsample_bicubic_2d_grad")                                     \
      .SetCreateFn<UpsampleBicubic2dGradCPUKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_UPSAMPLE_BICUBIC_CPU_KERNEL(float)
REGISTER_UPSAMPLE_BICUBIC_CPU_KERNEL(double)

}  // namespace oneflow
