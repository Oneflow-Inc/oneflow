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
__global__ void AdaptiveAvgPool2dGradCudaKernel(T* gradInput, T* gradOutput, int isizeH, int isizeW,
                                                int osizeH, int osizeW) {
  // iterators on input pixels
  int ih, iw;

  // select input/output plane based on thread/block ID
  int i_plane = blockIdx.x;
  int o_plane = i_plane;

  gradOutput = gradOutput + o_plane * osizeH * osizeW;
  gradInput = gradInput + i_plane * isizeH * isizeW;

  int istartH = blockDim.y * blockIdx.y + threadIdx.y;
  int iendH = isizeH;
  int istepH = blockDim.y * gridDim.y;

  int istartW = threadIdx.x;
  int iendW = isizeW;
  int istepW = blockDim.x;

  // compute gradInput
  for (ih = istartH; ih < iendH; ih += istepH) {
    int ostartH = START_IND(ih, isizeH, osizeH);
    int oendH = END_IND(ih, isizeH, osizeH);

    for (iw = istartW; iw < iendW; iw += istepW) {
      int ostartW = START_IND(iw, isizeW, osizeW);
      int oendW = END_IND(iw, isizeW, osizeW);

      // Compute the gradients over corresponding output pixels
      T* ptr_gradInput = gradInput + ih * isizeW + iw;

      int oh, ow;
      for (oh = ostartH; oh < oendH; ++oh) {
        int kH = START_IND(oh, osizeH, isizeH) - END_IND(oh, osizeH, isizeH);
        for (ow = ostartW; ow < oendW; ++ow) {
          int kW = START_IND(ow, osizeW, isizeW) - END_IND(ow, osizeW, isizeW);
          T grad_delta = gradOutput[ow + oh * osizeW] / kH / kW;
          *ptr_gradInput += grad_delta;
        }
      }
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

// __global__ void AdaptiveAvgPool2dCudaKernel(const T* input, T* output, int num_elems, int in_h,
//   int in_w, int out_h, int out_w) {

// template<>
// void GpuAdaptiveAvgPool2dFunctor<float16>::operator()(DeviceCtx* ctx, float16* in_ptr,
//                                                       float16* out_ptr, int isizeH, int isizeW,
//                                                       int osizeH, int osizeW, int64_t istrideD,
//                                                       int64_t istrideH, int64_t istrideW) {
//   RUN_CUDA_KERNEL((AdaptiveAvgPool2dCudaKernel<half>), ctx, reinterpret_cast<half*>(in_ptr),
//                   reinterpret_cast<half*>(out_ptr), isizeH, isizeW, osizeH, osizeW, istrideD,
//                   istrideH, istrideW);
// }

template<typename T>
struct GpuAdaptiveAvgpool2dGradFunctor final {
  void operator()(DeviceCtx* ctx, T* gradInput, T* gradOutput, int isizeH, int isizeW, int osizeH,
                  int osizeW) {
    // RUN_CUDA_KERNEL((AdaptiveAvgPool2dGradCudaKernel<T>), ctx, gradInput, gradOutput, isizeH,
    //                 isizeW, osizeH, osizeW);
  }
};

// template<>
// void GpuAdaptiveAvgpool2dGradFunctor<float16>::operator()(DeviceCtx* ctx, float16* gradInput,
//                                                           float16* gradOutput, int isizeH,
//                                                           int isizeW, int osizeH, int osizeW) {
//   RUN_CUDA_KERNEL((AdaptiveAvgPool2dGradCudaKernel<half>), ctx,
//   reinterpret_cast<half*>(gradInput),
//                   reinterpret_cast<half*>(gradOutput), isizeH, isizeW, osizeH, osizeW);
// }

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

    int n_idx = 0;
    int c_idx = 1;
    int h_idx = 2;
    int w_idx = 3;

    int in_h = in_tensor->shape().At(h_idx);
    int in_w = in_tensor->shape().At(w_idx);
    int out_h = out_tensor->shape().At(h_idx);
    int out_w = out_tensor->shape().At(w_idx);

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
// REGISTER_GPU_ADAPTIVE_AVGPOOL2D_KERNEL(DeviceType::kGPU, float16);

// template<DeviceType device_type, typename T>
// class GpuAdaptiveAvgPool2dGradKernel final : public OpKernel {
//  public:
//   GpuAdaptiveAvgPool2dGradKernel() = default;
//   ~GpuAdaptiveAvgPool2dGradKernel() = default;

//  private:
//   void Compute(KernelComputeContext* ctx) const override {
//     user_op::Tensor* grad_output = ctx->Tensor4ArgNameAndIndex("dy", 0);
//     user_op::Tensor* grad_input = ctx->Tensor4ArgNameAndIndex("dx", 0);
//     T* out_ptr = grad_output->dptr<T>();
//     T* in_ptr = grad_input->mut_dptr<T>();

//     const int64_t ndims = grad_output->shape().NumAxes();
//     CHECK_EQ(ndims, 4);

//     int64_t n_idx = 0;
//     int64_t c_idx = 1;
//     int64_t h_idx = 2;
//     int64_t w_idx = 3;

//     int isizeH = grad_input->shape().At(h_idx);
//     int isizeW = grad_input->shape().At(w_idx);
//     int osizeH = grad_output->shape().At(h_idx);
//     int osizeW = grad_output->shape().At(w_idx);

//     GpuAdaptiveAvgpool2dGradFunctor<T>()(ctx->device_ctx(), in_ptr, out_ptr, isizeH, isizeW,
//     osizeH,
//                                          osizeW);
//   }
//   bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
// };

// #define REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(device, dtype) \
//   REGISTER_USER_KERNEL("adaptive_avg_pool2d_grad")                     \
//       .SetCreateFn<GpuAdaptiveAvgPool2dGradKernel<device, dtype>>()    \
//       .SetIsMatchedHob((HobDeviceTag() == device)                      \
//                        & (HobDataType("dx", 0) == GetDataType<dtype>::value));

// REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(DeviceType::kGPU, float);
// REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(DeviceType::kGPU, double);
// REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(DeviceType::kGPU, float16);

}  // namespace user_op

}  // namespace oneflow
