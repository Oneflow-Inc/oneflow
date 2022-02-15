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
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
struct ReluGradByXFunctor {
  __device__ T Compute(T x, T dy) const {
    T zero_val = static_cast<T>(0.0); 
    if(x > zero_val){
        return dy; 
    }else{
        return zero_val;
    }
  }
};

template<typename FUNCTOR, typename T, typename Index>
__global__ void FusedBiasAddGradGpu(FUNCTOR grad_functor, const Index elem_cnt,
                                    const Index bias_size, const T* x,
                                    const T* bias, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    T x_i = x[i] + bias[(i % bias_size)];
    dx[i] = grad_functor.Compute(x_i, dy[i]);
  }
}


template<typename FUNCTOR, typename T>
void DispatchFusedBiasAddBackwardImpl(ep::Stream* stream, FUNCTOR grad_functor, int64_t elem_cnt,
                                      int64_t bias_size, 
                                      const T* x, const T* bias, const T* dy, T* dx) {
  if (IsKernelSafeInt32(elem_cnt)) {
    FusedBiasAddGradGpu<FUNCTOR, T, int32_t><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock,
                                            0, stream->As<ep::CudaStream>()->cuda_stream()>>>(grad_functor, elem_cnt, bias_size, x, bias, dy, dx); 
  } else {
    FusedBiasAddGradGpu<FUNCTOR, T, int64_t><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock,
                                            0, stream->As<ep::CudaStream>()->cuda_stream()>>>(grad_functor, elem_cnt, bias_size, x, bias, dy, dx); 
  }
}

}  // namespace


template<typename T>
class FusedMatmulBiasAddReluGradKernel final : public user_op::OpKernel {
 public:
  FusedMatmulBiasAddReluGradKernel() = default;
  ~FusedMatmulBiasAddReluGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* last_mlp_x = ctx->Tensor4ArgNameAndIndex("last_mlp_x", 0);
    const user_op::Tensor* last_mlp_bias = ctx->Tensor4ArgNameAndIndex("last_mlp_bias", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const DataType data_type = dy_tensor->data_type();
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* dbias_tensor = ctx->Tensor4ArgNameAndIndex("dbias", 0);

    const int64_t bias_size = last_mlp_bias->shape().At(0);
    const int64_t elem_cnt = last_mlp_x->shape().elem_cnt();
    ReluGradByXFunctor<T> relu_grad_by_x_functor;
    DispatchFusedBiasAddBackwardImpl<decltype(relu_grad_by_x_functor), T>(ctx->stream(), relu_grad_by_x_functor, elem_cnt,
        bias_size, last_mlp_x->dptr<T>(), last_mlp_bias->dptr<T>(), dy_tensor->dptr<T>(), dx_tensor->mut_dptr<T>()); 
    
    // Use Gemm to reduce bias_grad. 
    T* reduce_tmp_buffer = tmp_buffer->mut_dptr<T>();
    const int32_t m = bias_size;
    const int32_t n = 1;
    const int32_t k = last_mlp_x->shape().At(0);
    
    std::unique_ptr<ep::primitive::Fill> fill =
        ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->stream()->device_type(),
                                                                data_type);
    CHECK(fill);
    fill->Launch(ctx->stream(), reduce_tmp_buffer, 1.0, k);
    std::unique_ptr<ep::primitive::Matmul> matmul =
        ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(
            ctx->stream()->device_type(), data_type, ep::primitive::BlasTransposeType::T,
            ep::primitive::BlasTransposeType::N);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, 1.0, dx_tensor->mut_dptr(), reduce_tmp_buffer, 0.0,
                   dbias_tensor->mut_dptr());
  
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MATMUL_BIAS_ADD_GELU_GRAD_KERNEL(dtype)                \
  REGISTER_USER_KERNEL("fused_matmul_bias_add_relu_backward")                     \
      .SetCreateFn<FusedMatmulBiasAddReluGradKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value))\
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) { \
        const int64_t batch_size = ctx->InputTensorDesc("last_mlp_x", 0).shape().At(0); \
        const int64_t tmp_buffer_size = GetCudaAlignedSize(batch_size*sizeof(dtype));    \
        return tmp_buffer_size;                                          \
      });

REGISTER_FUSED_MATMUL_BIAS_ADD_GELU_GRAD_KERNEL(float)
REGISTER_FUSED_MATMUL_BIAS_ADD_GELU_GRAD_KERNEL(double)
// REGISTER_FUSED_MATMUL_BIAS_ADD_GELU_GRAD_KERNEL(half)

}  // namespace oneflow
