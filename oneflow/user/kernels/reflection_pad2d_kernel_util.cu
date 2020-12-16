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
#include <cstdint>
#ifdef WITH_CUDA
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/reflection_pad2d_kernel_util.h"

namespace oneflow {
namespace user_op {


template<typename T>
__global__ void DoCUDAReflectionPad2d(
    const T* src, T * dest,
    int64_t n_batch, int64_t n_channel,int64_t y_height, int64_t y_width,
    int64_t x_height, int64_t x_width, int64_t pad_left, int64_t pad_top
) {
  DoReflectionPad2d<T>(
    src, dest, n_batch, n_channel, y_height, y_width, 
    x_height, x_width, pad_left, pad_top
  );
};


template<typename T>
__global__ void DoCUDAReflectionPad2dGrad(
    const T* src, T * dest,
    int64_t n_batch, int64_t n_channel,int64_t dy_height, int64_t dy_width,
    int64_t dx_height, int64_t dx_width, int64_t pad_left, int64_t pad_top
) {
  DoReflectionPad2dGrad<T>(
    src, dest, n_batch, n_channel, dy_height, dy_width, 
    dx_height, dx_width, pad_left, pad_top
  );
};


template<typename T>
struct ReflectionPad2dFunctor<DeviceType::kGPU, T> final {
  void operator()(
      DeviceCtx* ctx, const T* src, T * dest,
      int64_t n_batch, int64_t n_channel,int64_t y_height, int64_t y_width,
      int64_t x_height, int64_t x_width, int64_t pad_left, int64_t pad_top
    ){
    int64_t elem_cnt = n_batch*n_channel*y_height*y_width;
    RUN_CUDA_KERNEL((DoCUDAReflectionPad2d<T>), ctx, BlocksNum4ThreadsNum(elem_cnt),
      src, dest, n_batch, n_channel, y_height, y_width, 
      x_height, x_width, pad_left, pad_top
    );
  }
};


// float16 special case of DimScatterAddFunctor template
template<>
void ReflectionPad2dFunctor<DeviceType::kGPU, float16>::operator()(
    DeviceCtx* ctx, const float16* src, float16 * dest,
      int64_t n_batch, int64_t n_channel,int64_t y_height, int64_t y_width,
      int64_t x_height, int64_t x_width, int64_t pad_left, int64_t pad_top
    ) {
      int64_t elem_cnt = n_batch*n_channel*y_height*y_width;
      RUN_CUDA_KERNEL((DoCUDAReflectionPad2d<half>), ctx, BlocksNum4ThreadsNum(elem_cnt),
          reinterpret_cast<const half*>(src), reinterpret_cast<half*>(dest), n_batch, n_channel, y_height, y_width, 
          x_height, x_width, pad_left, pad_top);
}


template<typename T>
struct ReflectionPad2dGradFunctor<DeviceType::kGPU, T> final {
  void operator()(
      DeviceCtx* ctx, const T* src, T * dest,
      int64_t n_batch, int64_t n_channel,int64_t dy_height, int64_t dy_width,
      int64_t dx_height, int64_t dx_width, int64_t pad_left, int64_t pad_top
    ){
    int64_t elem_cnt = n_batch*n_channel*dy_height*dy_width;
    RUN_CUDA_KERNEL((DoCUDAReflectionPad2dGrad<T>), ctx, BlocksNum4ThreadsNum(elem_cnt),
      src, dest, n_batch, n_channel, dy_height, dy_width, 
      dx_height, dx_width, pad_left, pad_top
    );
  }
};


template<>
void ReflectionPad2dGradFunctor<DeviceType::kGPU, float16>::operator()(
      DeviceCtx* ctx, const float16* src, float16 * dest,
      int64_t n_batch, int64_t n_channel,int64_t dy_height, int64_t dy_width,
      int64_t dx_height, int64_t dx_width, int64_t pad_left, int64_t pad_top
    ){
    int64_t elem_cnt = n_batch*n_channel*dy_height*dy_width;
    RUN_CUDA_KERNEL((DoCUDAReflectionPad2dGrad<half>), ctx, BlocksNum4ThreadsNum(elem_cnt),
      reinterpret_cast<const half*>(src), reinterpret_cast<half*>(dest), n_batch, n_channel, dy_height, dy_width, 
      dx_height, dx_width, pad_left, pad_top
    );
}


OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_REFLECTION_PAD2D_FUNCTOR,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 REFLECTION_PAD2D_DATA_TYPE_GPU_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_REFLECTION_PAD2D_GRAD_FUNCTOR,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 REFLECTION_PAD2D_GRAD_DATA_TYPE_GPU_SEQ);


}  // namespace user_op
}  // namespace oneflow

#endif  // WITH_CUDA