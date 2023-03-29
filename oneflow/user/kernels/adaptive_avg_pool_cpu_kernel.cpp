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
#include "oneflow/core/common/data_type.h"
#include "oneflow/user/kernels/adaptive_pool_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename accT>
void AvgForwardCompute(user_op::KernelComputeContext* ctx, const int32_t& dim) {
  user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
  user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const Shape& y_shape = ctx->TensorDesc4ArgNameAndIndex("y", 0)->shape();

  // TODO (Tianyu): Support 'channels_last'
  std::string data_format = "channels_first";
  const Shape& in = GetShape5D(x_shape, data_format, dim);
  const Shape& out = GetShape5D(y_shape, data_format, dim);

  const T* in_ptr = in_tensor->dptr<T>();
  T* out_ptr = out_tensor->mut_dptr<T>();

  const int64_t input_width = in.Count(4);
  const int64_t output_width = out.Count(4);
  const int64_t input_image_size = in.Count(3);
  const int64_t output_image_size = out.Count(3);
  const int64_t input_size = in.Count(2);
  const int64_t output_size = out.Count(2);

  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, c, 0, in.At(1)) {
      FOR_RANGE(int64_t, od, 0, out.At(2)) {
        int64_t id0 = start_index(od, out.At(2), in.At(2));
        int64_t id1 = end_index(od, out.At(2), in.At(2));
        int64_t kd = id1 - id0;
        FOR_RANGE(int64_t, oh, 0, out.At(3)) {
          int64_t ih0 = start_index(oh, out.At(3), in.At(3));
          int64_t ih1 = end_index(oh, out.At(3), in.At(3));
          int64_t kh = ih1 - ih0;
          FOR_RANGE(int64_t, ow, 0, out.At(4)) {
            int64_t iw0 = start_index(ow, out.At(4), in.At(4));
            int64_t iw1 = end_index(ow, out.At(4), in.At(4));
            int64_t kw = iw1 - iw0;

            // Compute local average
            accT sum = static_cast<accT>(0);
            FOR_RANGE(int64_t, id, id0, id1) {
              FOR_RANGE(int64_t, ih, ih0, ih1) {
                FOR_RANGE(int64_t, iw, iw0, iw1) {
                  sum += static_cast<accT>(in_ptr[id * input_image_size + ih * input_width + iw]);
                }
              }
            }
            out_ptr[od * output_image_size + oh * output_width + ow] =
                static_cast<T>(sum / kd / kh / kw);
          }
        }
      }
      in_ptr += input_size;
      out_ptr += output_size;
    }
  }
}

template<typename T>
void AvgBackwardCompute(user_op::KernelComputeContext* ctx, const int32_t& dim) {
  user_op::Tensor* grad_input = ctx->Tensor4ArgNameAndIndex("dx", 0);
  const user_op::Tensor* grad_output = ctx->Tensor4ArgNameAndIndex("dy", 0);
  const Shape& dx_shape = ctx->TensorDesc4ArgNameAndIndex("dx", 0)->shape();
  const Shape& dy_shape = ctx->TensorDesc4ArgNameAndIndex("dy", 0)->shape();

  // TODO (Tianyu): Support 'channels_last'
  std::string data_format = "channels_first";
  const Shape& in = GetShape5D(dx_shape, data_format, dim);
  const Shape& out = GetShape5D(dy_shape, data_format, dim);

  const T* out_ptr = grad_output->dptr<T>();
  T* in_ptr = grad_input->mut_dptr<T>();

  std::fill(in_ptr, in_ptr + grad_input->shape_view().elem_cnt(), static_cast<T>(0));

  const int64_t input_width = in.Count(4);
  const int64_t output_width = out.Count(4);
  const int64_t input_image_size = in.Count(3);
  const int64_t output_image_size = out.Count(3);
  const int64_t input_size = in.Count(2);
  const int64_t output_size = out.Count(2);

  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, c, 0, in.At(1)) {
      FOR_RANGE(int64_t, od, 0, out.At(2)) {
        int64_t id0 = start_index(od, out.At(2), in.At(2));
        int64_t id1 = end_index(od, out.At(2), in.At(2));
        int64_t kd = id1 - id0;
        FOR_RANGE(int64_t, oh, 0, out.At(3)) {
          int64_t ih0 = start_index(oh, out.At(3), in.At(3));
          int64_t ih1 = end_index(oh, out.At(3), in.At(3));
          int64_t kh = ih1 - ih0;
          FOR_RANGE(int64_t, ow, 0, out.At(4)) {
            int64_t iw0 = start_index(ow, out.At(4), in.At(4));
            int64_t iw1 = end_index(ow, out.At(4), in.At(4));
            int64_t kw = iw1 - iw0;
            T grad_delta = static_cast<T>(out_ptr[od * output_image_size + oh * output_width + ow]
                                          / kd / kh / kw);
            FOR_RANGE(int64_t, id, id0, id1) {
              FOR_RANGE(int64_t, ih, ih0, ih1) {
                FOR_RANGE(int64_t, iw, iw0, iw1) {
                  in_ptr[id * input_image_size + ih * input_width + iw] += grad_delta;
                }
              }
            }
          }
        }
      }
      in_ptr += input_size;
      out_ptr += output_size;
    }
  }
}
}  // namespace

template<DeviceType device_type, typename T>
class AdaptivePool1DCpuKernel final : public user_op::OpKernel {
 public:
  AdaptivePool1DCpuKernel() = default;
  ~AdaptivePool1DCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    if (GetDataType<T>::value == kFloat16) {
      AvgForwardCompute<T, float>(ctx, 1);
    } else {
      AvgForwardCompute<T, T>(ctx, 1);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class AdaptivePool2DCpuKernel final : public user_op::OpKernel {
 public:
  AdaptivePool2DCpuKernel() = default;
  ~AdaptivePool2DCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    if (GetDataType<T>::value == kFloat16) {
      AvgForwardCompute<T, float>(ctx, 2);
    } else {
      AvgForwardCompute<T, T>(ctx, 2);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class AdaptivePool3DCpuKernel final : public user_op::OpKernel {
 public:
  AdaptivePool3DCpuKernel() = default;
  ~AdaptivePool3DCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    if (GetDataType<T>::value == kFloat16) {
      AvgForwardCompute<T, float>(ctx, 3);
    } else {
      AvgForwardCompute<T, T>(ctx, 3);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
template<DeviceType device_type, typename T>
class AdaptivePool1DCpuGradKernel final : public user_op::OpKernel {
 public:
  AdaptivePool1DCpuGradKernel() = default;
  ~AdaptivePool1DCpuGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override { AvgBackwardCompute<T>(ctx, 1); }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class AdaptivePool2DCpuGradKernel final : public user_op::OpKernel {
 public:
  AdaptivePool2DCpuGradKernel() = default;
  ~AdaptivePool2DCpuGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override { AvgBackwardCompute<T>(ctx, 2); }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
template<DeviceType device_type, typename T>
class AdaptivePool3DCpuGradKernel final : public user_op::OpKernel {
 public:
  AdaptivePool3DCpuGradKernel() = default;
  ~AdaptivePool3DCpuGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override { AvgBackwardCompute<T>(ctx, 3); }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_POOL_KERNEL(device, dtype)                                    \
  REGISTER_USER_KERNEL("adaptive_avg_pool1d")                                           \
      .SetCreateFn<AdaptivePool1DCpuKernel<device, dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d")                                           \
      .SetCreateFn<AdaptivePool2DCpuKernel<device, dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("adaptive_avg_pool3d")                                           \
      .SetCreateFn<AdaptivePool3DCpuKernel<device, dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#define REGISTER_ADAPTIVE_POOL_KERNEL_WITH_DEVICE(device) \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, float16)          \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, float)            \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, double)           \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, int)

REGISTER_ADAPTIVE_POOL_KERNEL_WITH_DEVICE(DeviceType::kCPU)

#define REGISTER_ADAPTIVE_POOL_BACKWARD_KERNEL(device, dtype)                            \
  REGISTER_USER_KERNEL("adaptive_avg_pool1d_grad")                                       \
      .SetCreateFn<AdaptivePool1DCpuGradKernel<device, dtype>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                              \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d_grad")                                       \
      .SetCreateFn<AdaptivePool2DCpuGradKernel<device, dtype>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                              \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("adaptive_avg_pool3d_grad")                                       \
      .SetCreateFn<AdaptivePool3DCpuGradKernel<device, dtype>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                              \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#define REGISTER_ADAPTIVE_POOL_BACKWARD_KERNEL_WITH_DEVICE(device) \
  REGISTER_ADAPTIVE_POOL_BACKWARD_KERNEL(device, float16)          \
  REGISTER_ADAPTIVE_POOL_BACKWARD_KERNEL(device, float)            \
  REGISTER_ADAPTIVE_POOL_BACKWARD_KERNEL(device, double)           \
  REGISTER_ADAPTIVE_POOL_BACKWARD_KERNEL(device, int)

REGISTER_ADAPTIVE_POOL_BACKWARD_KERNEL_WITH_DEVICE(DeviceType::kCPU)
}  // namespace oneflow
