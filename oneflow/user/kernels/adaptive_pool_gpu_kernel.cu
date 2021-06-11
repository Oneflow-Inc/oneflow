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
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

namespace user_op {

template<typename T>
__global__ void AdaptiveAvgPool2dCudaKernel(T* input, T* output, int isizeH, int isizeW,
                          int osizeH, int osizeW,
                          int64_t istrideD, int64_t istrideH, int64_t istrideW) {
   // iterators on output pixels
    int oh, ow;

    // select input/output plane based on thread/block ID
    int o_plane = blockIdx.x;
    int i_plane = o_plane;

    output = output + o_plane*osizeH*osizeW;
    input = input + i_plane*istrideD;

    int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
    int oendH = osizeH;
    const int ostepH = blockDim.y*gridDim.y;

    int ostartW = threadIdx.x;
    int oendW = osizeW;
    const int ostepW = blockDim.x;

    // For all output pixels...
    for(oh = ostartH; oh < oendH; oh += ostepH) {

      int istartH = START_IND(oh, osizeH, isizeH);
      int iendH   = END_IND(oh, osizeH, isizeH);
      int kH = iendH - istartH;

      for(ow = ostartW; ow < oendW; ow += ostepW) {

        int istartW = START_IND(ow, osizeW, isizeW);
        int iendW   = END_IND(ow, osizeW, isizeW);
        int kW = iendW - istartW;

        // Compute the average pooling over corresponding input pixels
        T *ptr_input = input + istartH*istrideH + istartW*istrideW;
        T *ptr_output = output + oh*osizeW + ow;
        T sum = ScalarConvert<int, T>::to(0);
        int ih, iw;
        for(ih = 0; ih < kH; ++ih) {
          for(iw = 0; iw < kW; ++iw) {
            T val = ptr_input[iw*istrideW];
            sum += val;
          }
          ptr_input += istrideH; // next input line
        }
        // Update output
        *ptr_output = sum / kH / kW;
      }
    }
}

template<typename T>
__global__ void AdaptiveAvgPool2dGradCudaKernel(T *gradInput, T *gradOutput,
                    int isizeH, int isizeW, int osizeH, int osizeW) {
   // iterators on input pixels
    int ih, iw;

    // select input/output plane based on thread/block ID
    int i_plane = blockIdx.x;
    int o_plane = i_plane;

    gradOutput = gradOutput + o_plane*osizeH*osizeW;
    gradInput = gradInput + i_plane*isizeH*isizeW;

    int istartH = blockDim.y*blockIdx.y + threadIdx.y;
    int iendH = isizeH;
    int istepH = blockDim.y*gridDim.y;

    int istartW = threadIdx.x;
    int iendW = isizeW;
    int istepW = blockDim.x;

    // compute gradInput
    for(ih = istartH; ih < iendH; ih += istepH) {

      int ostartH = START_IND(ih, isizeH, osizeH);
      int oendH   = END_IND(ih, isizeH, osizeH);

      for(iw = istartW; iw < iendW; iw += istepW) {

        int ostartW = START_IND(iw, isizeW, osizeW);
        int oendW   = END_IND(iw, isizeW, osizeW);

        // Compute the gradients over corresponding output pixels
        T *ptr_gradInput = gradInput + ih*isizeW + iw;

        int oh, ow;
        for(oh = ostartH; oh < oendH; ++oh) {
          int kH = START_IND(oh, osizeH, isizeH) - END_IND(oh, osizeH, isizeH);
          for(ow = ostartW; ow < oendW; ++ow) {
            int kW = START_IND(ow, osizeW, isizeW) - END_IND(ow, osizeW, isizeW);
            T grad_delta = gradOutput[ow + oh*osizeW] / kH / kW;
            *ptr_gradInput += grad_delta;
          }
        }
      }
    }
}

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

    const int64_t n_idx = 0;
    const int64_t c_idx = 1;
    const int64_t h_idx = 2;
    const int64_t w_idx = 3;

    const int isizeH = in_tensor->shape().At(h_idx);
    const int isizeW = in_tensor->shape().At(w_idx);
    const int osizeH = out_tensor->shape().At(h_idx);
    const int osizeW = out_tensor->shape().At(w_idx);


  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_ADAPTIVE_AVGPOOL2D_KERNEL(device, dtype)   \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d")                   \
      .SetCreateFn<GpuAdaptiveAvgPool2dKernel<device, dtype>>() \
      .SetIsMatchedHob((HobDeviceTag() == device)               \
                       & (HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_GPU_ADAPTIVE_AVGPOOL2D_KERNEL(DeviceType::kGPU, float16);
REGISTER_GPU_ADAPTIVE_AVGPOOL2D_KERNEL(DeviceType::kGPU, float);
REGISTER_GPU_ADAPTIVE_AVGPOOL2D_KERNEL(DeviceType::kGPU, double);
REGISTER_GPU_ADAPTIVE_AVGPOOL2D_KERNEL(DeviceType::kGPU, int);

template<DeviceType device_type, typename T>
class GpuEluGradKernel final : public OpKernel {
 public:
  GpuEluGradKernel() = default;
  ~GpuEluGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(device, dtype) \
  REGISTER_USER_KERNEL("elu_grad")                                     \
      .SetCreateFn<GpuEluGradKernel<device, dtype>>()                  \
      .SetIsMatchedHob((HobDeviceTag() == device)                      \
                       & (HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(DeviceType::kGPU, float16);
REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(DeviceType::kGPU, float);
REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(DeviceType::kGPU, double);
REGISTER_GPU_ADAPTIVE_AVGPOOL2D_BACKWARD_KERNEL(DeviceType::kGPU, int);

}  // namespace user_op

}  // namespace oneflow
