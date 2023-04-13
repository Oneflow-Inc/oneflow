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
__global__ void AdaptiveMaxPoolCudaKernel(const T* input, T* output, int64_t* return_index,
                                          int num_elems, int in_d, int in_h, int in_w, int out_d,
                                          int out_h, int out_w) {
  const int out_panel_size = out_d * out_h * out_w;
  const int in_panel_size = in_d * in_h * in_w;
  const int out_hw = out_w * out_h;

  CUDA_1D_KERNEL_LOOP(idx, num_elems) {
    int bc_idx = idx / out_panel_size;
    int out_d_idx = (idx % out_panel_size) / out_hw;
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

    int64_t batch_idx_base = bc_idx * in_panel_size;
    const T* in_ptr =
        input + batch_idx_base + in_start_d * in_h * in_w + in_start_h * in_w + in_start_w;
    T local_max = in_ptr[0];
    int64_t local_max_index = static_cast<int64_t>(in_ptr - input) - batch_idx_base;
    for (int id = 0; id < k_d; ++id) {
      for (int ih = 0; ih < k_h; ++ih) {
        for (int iw = 0; iw < k_w; ++iw) {
          T val = *(in_ptr + ih * in_w + iw);
          if (val > local_max) {
            local_max = val;
            local_max_index = in_ptr - input - batch_idx_base + ih * in_w + iw;
          }
        }
      }
      in_ptr += in_h * in_w;  // next input depth
    }

    output[idx] = local_max;
    return_index[idx] = local_max_index;
  }
}

template<typename T>
__global__ void AdaptiveMaxPoolGradCudaKernel(T* input, const T* output, const int64_t* index,
                                              int dy_elems, int in_panel_size, int out_panel_size) {
  CUDA_1D_KERNEL_LOOP(idx, dy_elems) {
    int bc_idx = idx / out_panel_size;
    T* input_ptr = input + bc_idx * in_panel_size;
    cuda::atomic::Add(input_ptr + index[idx], output[idx]);
  }
}

template<typename T, int32_t dim>
void MaxForwardCompute(KernelComputeContext* ctx) {
  const Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
  Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
  Tensor* return_indices = ctx->Tensor4ArgNameAndIndex("index", 0);

  const T* in_ptr = in_tensor->dptr<T>();
  T* out_ptr = out_tensor->mut_dptr<T>();
  int64_t* index_ptr = return_indices->mut_dptr<int64_t>();

  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const Shape& y_shape = ctx->TensorDesc4ArgNameAndIndex("y", 0)->shape();

  // TODO: Support 'channels_last'
  std::string data_format = "channels_first";
  const Shape& in = GetShape5D(x_shape, data_format, dim);
  const Shape& out = GetShape5D(y_shape, data_format, dim);

  const int out_elems = out_tensor->shape_view().elem_cnt();

  RUN_CUDA_KERNEL((AdaptiveMaxPoolCudaKernel<T>), ctx->stream(), out_elems, in_ptr, out_ptr,
                  index_ptr, out_elems, in.At(2), in.At(3), in.At(4), out.At(2), out.At(3),
                  out.At(4));
}

template<typename T, int32_t dim>
void MaxBackwardCompute(KernelComputeContext* ctx) {
  const Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
  Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
  const user_op::Tensor* return_indices = ctx->Tensor4ArgNameAndIndex("index", 0);

  const T* out_ptr = out_tensor->dptr<T>();
  T* in_ptr = in_tensor->mut_dptr<T>();
  const int64_t* index_ptr = return_indices->dptr<int64_t>();

  const Shape& dx_shape = ctx->TensorDesc4ArgNameAndIndex("dx", 0)->shape();
  const Shape& dy_shape = ctx->TensorDesc4ArgNameAndIndex("dy", 0)->shape();

  // TODO (Tianyu): Support 'channels_last'
  std::string data_format = "channels_first";
  const Shape& in = GetShape5D(dx_shape, data_format, dim);
  const Shape& out = GetShape5D(dy_shape, data_format, dim);

  const int in_elems = in_tensor->shape_view().elem_cnt();
  const int out_elems = out_tensor->shape_view().elem_cnt();

  std::unique_ptr<ep::primitive::Memset> memset_primitive =
      ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
  CHECK(memset_primitive);
  memset_primitive->Launch(ctx->stream(), in_ptr, 0, in_elems * sizeof(T));
  RUN_CUDA_KERNEL((AdaptiveMaxPoolGradCudaKernel<T>), ctx->stream(), out_elems, in_ptr, out_ptr,
                  index_ptr, out_elems, in.At(2) * in.At(3) * in.At(4),
                  out.At(2) * out.At(3) * out.At(4));
}

template<DeviceType device_type, typename T, int32_t dim>
class GpuAdaptiveMaxPoolNdKernel final : public OpKernel {
 public:
  GpuAdaptiveMaxPoolNdKernel() = default;
  ~GpuAdaptiveMaxPoolNdKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override { MaxForwardCompute<T, dim>(ctx); }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T, int32_t dim>
class GpuAdaptiveMaxPoolNdGradKernel final : public OpKernel {
 public:
  GpuAdaptiveMaxPoolNdGradKernel() = default;
  ~GpuAdaptiveMaxPoolNdGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override { MaxBackwardCompute<T, dim>(ctx); }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_ADAPTIVE_MAXPOOL_KERNEL(device, dtype)                   \
  REGISTER_USER_KERNEL("adaptive_max_pool1d")                                  \
      .SetCreateFn<GpuAdaptiveMaxPoolNdKernel<device, dtype, 1>>()             \
      .SetIsMatchedHob((HobDeviceType() == device)                             \
                       && (HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("adaptive_max_pool2d")                                  \
      .SetCreateFn<GpuAdaptiveMaxPoolNdKernel<device, dtype, 2>>()             \
      .SetIsMatchedHob((HobDeviceType() == device)                             \
                       && (HobDataType("y", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("adaptive_max_pool3d")                                  \
      .SetCreateFn<GpuAdaptiveMaxPoolNdKernel<device, dtype, 3>>()             \
      .SetIsMatchedHob((HobDeviceType() == device)                             \
                       && (HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_ADAPTIVE_MAXPOOL_KERNEL(DeviceType::kCUDA, float);
REGISTER_CUDA_ADAPTIVE_MAXPOOL_KERNEL(DeviceType::kCUDA, double);
REGISTER_CUDA_ADAPTIVE_MAXPOOL_KERNEL(DeviceType::kCUDA, int);

#define REGISTER_CUDA_ADAPTIVE_MAXPOOL_BACKWARD_KERNEL(device, dtype)           \
  REGISTER_USER_KERNEL("adaptive_max_pool1d_grad")                              \
      .SetCreateFn<GpuAdaptiveMaxPoolNdGradKernel<device, dtype, 1>>()          \
      .SetIsMatchedHob((HobDeviceType() == device)                              \
                       && (HobDataType("dx", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("adaptive_max_pool2d_grad")                              \
      .SetCreateFn<GpuAdaptiveMaxPoolNdGradKernel<device, dtype, 2>>()          \
      .SetIsMatchedHob((HobDeviceType() == device)                              \
                       && (HobDataType("dx", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("adaptive_max_pool3d_grad")                              \
      .SetCreateFn<GpuAdaptiveMaxPoolNdGradKernel<device, dtype, 3>>()          \
      .SetIsMatchedHob((HobDeviceType() == device)                              \
                       && (HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_ADAPTIVE_MAXPOOL_BACKWARD_KERNEL(DeviceType::kCUDA, float);
REGISTER_CUDA_ADAPTIVE_MAXPOOL_BACKWARD_KERNEL(DeviceType::kCUDA, double);
REGISTER_CUDA_ADAPTIVE_MAXPOOL_BACKWARD_KERNEL(DeviceType::kCUDA, int);

}  // namespace user_op

}  // namespace oneflow
