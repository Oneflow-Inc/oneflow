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

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace oneflow {

namespace user_op {

#define START_IND(a, b, c) (int)std::floor((float)(a * c) / b)
#define END_IND(a, b, c) (int)std::ceil((float)((a + 1) * c) / b)

#define START_IND_INT(a, b, c) ((a * c) / b)
#define END_IND_INT(a, b, c) (((a + 1) * c + b - 1) / b)

template<typename T>
__global__ void InitPtr(int elements, T* ptr) {
  int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  while (gid < elements) {
    ptr[gid] = static_cast<T>(0);
    gid += step;
  }
}

template<typename T>
__global__ void AdaptiveAvgPool2dCudaKernel(const T* input, T* output, int num_elems, int in_h,
                                            int in_w, int out_h, int out_w) {
  const int out_panel_size = out_h * out_w;
  const int in_panel_size = in_h * in_w;

  CUDA_1D_KERNEL_LOOP(idx, num_elems) {
    int bc_idx = idx / out_panel_size;
    int out_h_idx = (idx % out_panel_size) / out_w;
    int out_w_idx = (idx % out_panel_size) % out_w;

    int in_start_h = START_IND(out_h_idx, out_h, in_h);
    int in_end_h = END_IND(out_h_idx, out_h, in_h);
    int k_h = in_end_h - in_start_h;

    int in_start_w = START_IND(out_w_idx, out_w, in_w);
    int in_end_w = END_IND(out_w_idx, out_w, in_w);
    int k_w = in_end_w - in_start_w;

    const T* in_ptr = input + bc_idx * in_panel_size + in_start_h * in_w + in_start_w;
    T sum = static_cast<T>(0);
    for (int ih = 0; ih < k_h; ++ih) {
      for (int iw = 0; iw < k_w; ++iw) {
        T val = in_ptr[iw];
        sum += val;
      }
      in_ptr += in_w;  // next input line
    }
    // Update output
    output[idx] = sum / k_h / k_w;
  }
}

template<typename T>
__global__ void AdaptiveAvgPool2dGradCudaKernel(T* input, const T* output, int num_elems, int in_h,
                                                int in_w, int out_h, int out_w) {
  const int out_panel_size = out_h * out_w;
  const int in_panel_size = in_h * in_w;

  CUDA_1D_KERNEL_LOOP(idx, num_elems) {
    int bc_idx = idx / out_panel_size;
    int out_h_idx = (idx % out_panel_size) / out_w;
    int out_w_idx = (idx % out_panel_size) % out_w;

    int in_start_h = START_IND(out_h_idx, out_h, in_h);
    int in_end_h = END_IND(out_h_idx, out_h, in_h);
    int k_h = in_end_h - in_start_h;

    int in_start_w = START_IND(out_w_idx, out_w, in_w);
    int in_end_w = END_IND(out_w_idx, out_w, in_w);
    int k_w = in_end_w - in_start_w;

    const T grad_delta = output[idx] / k_h / k_w;
    T* input_ptr = input + bc_idx * in_panel_size + in_start_h * in_w + in_start_w;
    for (int ih = 0; ih < k_h; ++ih) {
      for (int iw = 0; iw < k_w; ++iw) { input_ptr[iw] += grad_delta; }
      input_ptr += in_w;
    }
  }
}

template<typename T>
struct GpuAdaptiveAvgPool2dFunctor final {
  void operator()(DeviceCtx* ctx, const T* input, T* output, int num_elems, int in_h, int in_w,
                  int out_h, int out_w) {
    RUN_CUDA_KERNEL((AdaptiveAvgPool2dCudaKernel<T>), ctx, num_elems, input, output, num_elems,
                    in_h, in_w, out_h, out_w);
  }
};

template<typename T>
struct GpuAdaptiveAvgpool2dGradFunctor final {
  void operator()(DeviceCtx* ctx, T* input, const T* output, int num_elems, int input_elems,
                  int in_h, int in_w, int out_h, int out_w) {
    RUN_CUDA_KERNEL((InitPtr<T>), ctx, input_elems, input_elems, input);
    RUN_CUDA_KERNEL((AdaptiveAvgPool2dGradCudaKernel<T>), ctx, num_elems, input, output, num_elems,
                    in_h, in_w, out_h, out_w);
  }
};

template<DeviceType device_type, typename T>
class GpuAdaptiveAvgPool2dKernel final : public OpKernel {
 public:
  GpuAdaptiveAvgPool2dKernel() = default;
  ~GpuAdaptiveAvgPool2dKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();

    const int ndims = in_tensor->shape().NumAxes();
    CHECK_EQ(ndims, 4);

    const int out_elems = out_tensor->shape().elem_cnt();

    const int h_idx = 2;
    const int w_idx = 3;

    const int in_h = in_tensor->shape().At(h_idx);
    const int in_w = in_tensor->shape().At(w_idx);
    const int out_h = out_tensor->shape().At(h_idx);
    const int out_w = out_tensor->shape().At(w_idx);

    GpuAdaptiveAvgPool2dFunctor<T>()(ctx->device_ctx(), in_ptr, out_ptr, out_elems, in_h, in_w,
                                     out_h, out_w);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_ADAPTIVE_AVGPOOL2D_KERNEL(device, dtype)   \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d")                   \
      .SetCreateFn<GpuAdaptiveAvgPool2dKernel<device, dtype>>() \
      .SetIsMatchedHob((HobDeviceTag() == device)               \
                       & (HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_GPU_ADAPTIVE_AVGPOOL2D_KERNEL(DeviceType::kGPU, float);
REGISTER_GPU_ADAPTIVE_AVGPOOL2D_KERNEL(DeviceType::kGPU, double);

template<DeviceType device_type, typename T>
class GpuAdaptiveAvgPool2dGradKernel final : public OpKernel {
 public:
  GpuAdaptiveAvgPool2dGradKernel() = default;
  ~GpuAdaptiveAvgPool2dGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const T* out_ptr = out_tensor->dptr<T>();
    T* in_ptr = in_tensor->mut_dptr<T>();

    const int64_t ndims = out_tensor->shape().NumAxes();
    CHECK_EQ(ndims, 4);

    const int in_elems = in_tensor->shape().elem_cnt();
    const int out_elems = out_tensor->shape().elem_cnt();

    const int64_t h_idx = 2;
    const int64_t w_idx = 3;

    const int in_h = in_tensor->shape().At(h_idx);
    const int in_w = in_tensor->shape().At(w_idx);
    const int out_h = out_tensor->shape().At(h_idx);
    const int out_w = out_tensor->shape().At(w_idx);

    GpuAdaptiveAvgpool2dGradFunctor<T>()(ctx->device_ctx(), in_ptr, out_ptr, out_elems, in_elems,
                                         in_h, in_w, out_h, out_w);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(device, dtype) \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d_grad")                     \
      .SetCreateFn<GpuAdaptiveAvgPool2dGradKernel<device, dtype>>()    \
      .SetIsMatchedHob((HobDeviceTag() == device)                      \
                       & (HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(DeviceType::kGPU, float);
REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(DeviceType::kGPU, double);

}  // namespace user_op

}  // namespace oneflow
