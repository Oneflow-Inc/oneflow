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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"

namespace oneflow {

namespace user_op {

template<typename T>
__global__ void AdaptiveAvgPool2dCudaKernel(T* input, T* output, int isizeH, int isizeW, int osizeH,
                                            int osizeW, int64_t istrideD, int64_t istrideH,
                                            int64_t istrideW) {
  // iterators on output pixels
  int oh, ow;

  // select input/output plane based on thread/block ID
  int o_plane = blockIdx.x;
  int i_plane = o_plane;

  output = output + o_plane * osizeH * osizeW;
  input = input + i_plane * istrideD;

  int ostartH = blockDim.y * blockIdx.y + threadIdx.y;
  int oendH = osizeH;
  const int ostepH = blockDim.y * gridDim.y;

  int ostartW = threadIdx.x;
  int oendW = osizeW;
  const int ostepW = blockDim.x;

  // For all output pixels...
  for (oh = ostartH; oh < oendH; oh += ostepH) {
    int istartH = START_IND(oh, osizeH, isizeH);
    int iendH = END_IND(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for (ow = ostartW; ow < oendW; ow += ostepW) {
      int istartW = START_IND(ow, osizeW, isizeW);
      int iendW = END_IND(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the average pooling over corresponding input pixels
      T* ptr_input = input + istartH * istrideH + istartW * istrideW;
      T* ptr_output = output + oh * osizeW + ow;
      T sum = static_cast<T>(0);
      int ih, iw;
      for (ih = 0; ih < kH; ++ih) {
        for (iw = 0; iw < kW; ++iw) {
          T val = ptr_input[iw * istrideW];
          sum += val;
        }
        ptr_input += istrideH;  // next input line
      }
      // Update output
      *ptr_output = sum / kH / kW;
    }
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
  void operator()(DeviceCtx* ctx, T* in_ptr, T* out_ptr, int isizeH, int isizeW, int osizeH,
                  int osizeW, int64_t istrideD, int64_t istrideH, int64_t istrideW) {
    RUN_CUDA_KERNEL((AdaptiveAvgPool2dCudaKernel<T>), ctx, in_ptr, out_ptr, isizeH, isizeW, osizeH,
                    osizeW, istrideD, istrideH, istrideW);
  }
};

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
    RUN_CUDA_KERNEL((AdaptiveAvgPool2dGradCudaKernel<T>), ctx, gradInput, gradOutput, isizeH,
                    isizeW, osizeH, osizeW);
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
    Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();

    const int64_t ndims = in_tensor->shape().NumAxes();
    CHECK_EQ(ndims, 4);

    int64_t n_idx = 0;
    int64_t c_idx = 1;
    int64_t h_idx = 2;
    int64_t w_idx = 3;

    int isizeH = in_tensor->shape().At(h_idx);
    int isizeW = in_tensor->shape().At(w_idx);
    int osizeH = out_tensor->shape().At(h_idx);
    int osizeW = out_tensor->shape().At(w_idx);

    int64_t istrideD = in_tensor->shape().At(c_idx);
    int64_t istrideH = isizeH;
    int64_t istrideW = isizeW;

    GpuAdaptiveAvgPool2dFunctor<T>()(ctx->device_ctx(), in_ptr, out_ptr, isizeH, isizeW, osizeH,
                                     osizeW, istrideD, istrideH, istrideW);
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

template<DeviceType device_type, typename T>
class GpuAdaptiveAvgPool2dGradKernel final : public OpKernel {
 public:
  GpuAdaptiveAvgPool2dGradKernel() = default;
  ~GpuAdaptiveAvgPool2dGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    user_op::Tensor* grad_output = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* grad_input = ctx->Tensor4ArgNameAndIndex("dx", 0);
    T* out_ptr = grad_output->dptr<T>();
    T* in_ptr = grad_input->mut_dptr<T>();

    const int64_t ndims = grad_output->shape().NumAxes();
    CHECK_EQ(ndims, 4);

    int64_t n_idx = 0;
    int64_t c_idx = 1;
    int64_t h_idx = 2;
    int64_t w_idx = 3;

    int isizeH = grad_input->shape().At(h_idx);
    int isizeW = grad_input->shape().At(w_idx);
    int osizeH = grad_output->shape().At(h_idx);
    int osizeW = grad_output->shape().At(w_idx);

    GpuAdaptiveAvgpool2dGradFunctor<T>()(ctx->device_ctx(), in_ptr, out_ptr, isizeH, isizeW, osizeH,
                                         osizeW);
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
// REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(DeviceType::kGPU, float16);

}  // namespace user_op

}  // namespace oneflow
