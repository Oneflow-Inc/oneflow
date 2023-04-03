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
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/user/utils/pool_util.h"
#include "oneflow/user/kernels/adaptive_pool_kernel_util.h"

namespace oneflow {

namespace user_op {

template<typename T>
__global__ void InitPtr(int elements, T* ptr) {
  int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  while (gid < elements) {
    ptr[gid] = static_cast<T>(0);
    gid += step;
  }
}

inline Shape GetShape5D(const Shape& shape, const std::string& data_format, int32_t dim) {
  FixedDimVector shape_3d = {GetInDim(shape, data_format, 0, dim),
                             GetInDim(shape, data_format, 1, dim),
                             GetInDim(shape, data_format, 2, dim)};
  return Shape({shape.At(0), shape.At(1), shape_3d.at(0), shape_3d.at(1), shape_3d.at(2)});
}

template<typename T>
__global__ void AdaptiveAvgPoolCudaKernel(const T* input, T* output, int num_elems, int in_d,
                                          int in_h, int in_w, int out_d, int out_h, int out_w) {
  const int out_panel_size = out_d * out_h * out_w;
  const int in_panel_size = in_d * in_h * in_w;

  CUDA_1D_KERNEL_LOOP(idx, num_elems) {
    // TODO (Tianyu): Replace following codes with 'NdIndexOffsetHelper'
    int bc_idx = idx / out_panel_size;
    int out_d_idx = (idx % out_panel_size) / out_w / out_h;
    int out_h_idx = (idx % out_panel_size) % (out_h * out_w) / out_w;
    int out_w_idx = (idx % out_panel_size) % (out_h * out_w) % out_w;

    int in_start_d = START_IND(out_d_idx, out_d, in_d);
    int in_end_d = END_IND(out_d_idx, out_d, in_d);
    int k_d = in_end_d - in_start_d;

    int in_start_h = START_IND(out_h_idx, out_h, in_h);
    int in_end_h = END_IND(out_h_idx, out_h, in_h);
    int k_h = in_end_h - in_start_h;

    int in_start_w = START_IND(out_w_idx, out_w, in_w);
    int in_end_w = END_IND(out_w_idx, out_w, in_w);
    int k_w = in_end_w - in_start_w;

    const T* in_ptr =
        input + bc_idx * in_panel_size + in_start_d * in_h * in_w + in_start_h * in_w + in_start_w;
    T sum = static_cast<T>(0);
    for (int id = 0; id < k_d; ++id) {
      for (int ih = 0; ih < k_h; ++ih) {
        for (int iw = 0; iw < k_w; ++iw) {
          T val = *(in_ptr + ih * in_w + iw);
          sum += val;
        }
      }
      in_ptr += in_h * in_w;  // next input depth
    }
    // Update output
    output[idx] = sum / static_cast<T>(k_d) / static_cast<T>(k_h) / static_cast<T>(k_w);
  }
}

template<typename T>
__global__ void AdaptiveAvgPoolGradCudaKernel(T* input, const T* output, int num_elems, int in_d,
                                              int in_h, int in_w, int out_d, int out_h, int out_w) {
  const int out_panel_size = out_d * out_h * out_w;
  const int in_panel_size = in_d * in_h * in_w;

  CUDA_1D_KERNEL_LOOP(idx, num_elems) {
    // TODO (Tianyu): Replace following codes with 'NdIndexOffsetHelper'
    int bc_idx = idx / out_panel_size;
    int out_d_idx = (idx % out_panel_size) / out_w / out_h;
    int out_h_idx = (idx % out_panel_size) % (out_h * out_w) / out_w;
    int out_w_idx = (idx % out_panel_size) % (out_h * out_w) % out_w;

    int in_start_d = START_IND(out_d_idx, out_d, in_d);
    int in_end_d = END_IND(out_d_idx, out_d, in_d);
    int k_d = in_end_d - in_start_d;

    int in_start_h = START_IND(out_h_idx, out_h, in_h);
    int in_end_h = END_IND(out_h_idx, out_h, in_h);
    int k_h = in_end_h - in_start_h;

    int in_start_w = START_IND(out_w_idx, out_w, in_w);
    int in_end_w = END_IND(out_w_idx, out_w, in_w);
    int k_w = in_end_w - in_start_w;

    const T grad_delta =
        output[idx] / static_cast<T>(k_d) / static_cast<T>(k_h) / static_cast<T>(k_w);
    T* input_ptr =
        input + bc_idx * in_panel_size + in_start_d * in_h * in_w + in_start_h * in_w + in_start_w;
    for (int id = 0; id < k_d; ++id) {
      for (int ih = 0; ih < k_h; ++ih) {
        for (int iw = 0; iw < k_w; ++iw) {
          // TODO (Tianyu): Use 'atmoic::Add' when necessary
          cuda::atomic::Add(input_ptr + ih * in_w + iw, grad_delta);
        }
      }
      input_ptr += in_h * in_w;  // next input depth
    }
  }
}

template<typename T>
void AvgForwardCompute(KernelComputeContext* ctx, const int32_t& dim) {
  const Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
  Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
  const T* in_ptr = in_tensor->dptr<T>();
  T* out_ptr = out_tensor->mut_dptr<T>();

  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const Shape& y_shape = ctx->TensorDesc4ArgNameAndIndex("y", 0)->shape();

  // TODO (Tianyu): Support 'channels_last'
  std::string data_format = "channels_first";
  const Shape& in = GetShape5D(x_shape, data_format, dim);
  const Shape& out = GetShape5D(y_shape, data_format, dim);

  const int out_elems = out_tensor->shape_view().elem_cnt();

  RUN_CUDA_KERNEL((AdaptiveAvgPoolCudaKernel<T>), ctx->stream(), out_elems, in_ptr, out_ptr,
                  out_elems, in.At(2), in.At(3), in.At(4), out.At(2), out.At(3), out.At(4));
}

template<typename T>
void AvgBackwardCompute(KernelComputeContext* ctx, const int32_t& dim) {
  const Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
  Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
  const T* out_ptr = out_tensor->dptr<T>();
  T* in_ptr = in_tensor->mut_dptr<T>();

  const Shape& dx_shape = ctx->TensorDesc4ArgNameAndIndex("dx", 0)->shape();
  const Shape& dy_shape = ctx->TensorDesc4ArgNameAndIndex("dy", 0)->shape();

  // TODO (Tianyu): Support 'channels_last'
  std::string data_format = "channels_first";
  const Shape& in = GetShape5D(dx_shape, data_format, dim);
  const Shape& out = GetShape5D(dy_shape, data_format, dim);

  const int in_elems = in_tensor->shape_view().elem_cnt();
  const int out_elems = out_tensor->shape_view().elem_cnt();

  RUN_CUDA_KERNEL((InitPtr<T>), ctx->stream(), in_elems, in_elems, in_ptr);
  RUN_CUDA_KERNEL((AdaptiveAvgPoolGradCudaKernel<T>), ctx->stream(), out_elems, in_ptr, out_ptr,
                  out_elems, in.At(2), in.At(3), in.At(4), out.At(2), out.At(3), out.At(4));
}

template<DeviceType device_type, typename T>
class GpuAdaptiveAvgPool1dKernel final : public OpKernel {
 public:
  GpuAdaptiveAvgPool1dKernel() = default;
  ~GpuAdaptiveAvgPool1dKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override { AvgForwardCompute<T>(ctx, 1); }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class GpuAdaptiveAvgPool2dKernel final : public OpKernel {
 public:
  GpuAdaptiveAvgPool2dKernel() = default;
  ~GpuAdaptiveAvgPool2dKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override { AvgForwardCompute<T>(ctx, 2); }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class GpuAdaptiveAvgPool3dKernel final : public OpKernel {
 public:
  GpuAdaptiveAvgPool3dKernel() = default;
  ~GpuAdaptiveAvgPool3dKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override { AvgForwardCompute<T>(ctx, 3); }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class GpuAdaptiveAvgPool1dGradKernel final : public OpKernel {
 public:
  GpuAdaptiveAvgPool1dGradKernel() = default;
  ~GpuAdaptiveAvgPool1dGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override { AvgBackwardCompute<T>(ctx, 1); }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class GpuAdaptiveAvgPool2dGradKernel final : public OpKernel {
 public:
  GpuAdaptiveAvgPool2dGradKernel() = default;
  ~GpuAdaptiveAvgPool2dGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override { AvgBackwardCompute<T>(ctx, 2); }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class GpuAdaptiveAvgPool3dGradKernel final : public OpKernel {
 public:
  GpuAdaptiveAvgPool3dGradKernel() = default;
  ~GpuAdaptiveAvgPool3dGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override { AvgBackwardCompute<T>(ctx, 3); }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_ADAPTIVE_AVGPOOL_KERNEL(device, dtype)                   \
  REGISTER_USER_KERNEL("adaptive_avg_pool1d")                                  \
      .SetCreateFn<GpuAdaptiveAvgPool1dKernel<device, dtype>>()                \
      .SetIsMatchedHob((HobDeviceType() == device)                             \
                       && (HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d")                                  \
      .SetCreateFn<GpuAdaptiveAvgPool2dKernel<device, dtype>>()                \
      .SetIsMatchedHob((HobDeviceType() == device)                             \
                       && (HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("adaptive_avg_pool3d")                                  \
      .SetCreateFn<GpuAdaptiveAvgPool3dKernel<device, dtype>>()                \
      .SetIsMatchedHob((HobDeviceType() == device)                             \
                       && (HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_ADAPTIVE_AVGPOOL_KERNEL(DeviceType::kCUDA, half);
REGISTER_CUDA_ADAPTIVE_AVGPOOL_KERNEL(DeviceType::kCUDA, float);
REGISTER_CUDA_ADAPTIVE_AVGPOOL_KERNEL(DeviceType::kCUDA, double);
REGISTER_CUDA_ADAPTIVE_AVGPOOL_KERNEL(DeviceType::kCUDA, int);

#define REGISTER_CUDA_ADAPTIVE_AVGPOOL_BACKWARD_KERNEL(device, dtype)           \
  REGISTER_USER_KERNEL("adaptive_avg_pool1d_grad")                              \
      .SetCreateFn<GpuAdaptiveAvgPool1dGradKernel<device, dtype>>()             \
      .SetIsMatchedHob((HobDeviceType() == device)                              \
                       && (HobDataType("dx", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d_grad")                              \
      .SetCreateFn<GpuAdaptiveAvgPool2dGradKernel<device, dtype>>()             \
      .SetIsMatchedHob((HobDeviceType() == device)                              \
                       && (HobDataType("dx", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("adaptive_avg_pool3d_grad")                              \
      .SetCreateFn<GpuAdaptiveAvgPool3dGradKernel<device, dtype>>()             \
      .SetIsMatchedHob((HobDeviceType() == device)                              \
                       && (HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_ADAPTIVE_AVGPOOL_BACKWARD_KERNEL(DeviceType::kCUDA, half);
REGISTER_CUDA_ADAPTIVE_AVGPOOL_BACKWARD_KERNEL(DeviceType::kCUDA, float);
REGISTER_CUDA_ADAPTIVE_AVGPOOL_BACKWARD_KERNEL(DeviceType::kCUDA, double);
REGISTER_CUDA_ADAPTIVE_AVGPOOL_BACKWARD_KERNEL(DeviceType::kCUDA, int);

}  // namespace user_op

}  // namespace oneflow
