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
    const Tensor*  x, Tensor* y, int64_t c_idx, int64_t h_idx, int64_t w_idx, int64_t pad_left, int64_t pad_top
) {
  printf("\n .cu >>>>>>>>>>>>>>>>>>> DoCUDAReflectionPad2d !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
  DoReflectionPad2d<T>(x, y, c_idx, h_idx, w_idx, pad_left, pad_top);
  printf("\n .cu >>>>>>>>>>>>>>>>>>> DoCUDAReflectionPad2d finished !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
};


template<typename T>
__global__ void DoCUDAReflectionPad2dGrad(
    const Tensor*  dy, Tensor* dx, int64_t c_idx, int64_t h_idx, int64_t w_idx, int64_t pad_left, int64_t pad_top
) {
  printf("\n .cu >>>>>>>>>>>>>>>>>>> DoCUDAReflectionPad2dGrad !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
  DoReflectionPad2dGrad<T>(dy, dx, c_idx, h_idx, w_idx, pad_left, pad_top);
};


template<typename T>
struct ReflectionPad2dFunctor<DeviceType::kGPU, T> final {
  void operator()(
      DeviceCtx* ctx, const Tensor*  x, Tensor* y, int64_t c_idx, int64_t h_idx, int64_t w_idx, int64_t pad_left, int64_t pad_top
    ){
    int64_t  elem_cnt = y->shape().elem_cnt();
    printf("\n.cu >>>>>>>>>>>>>>>>> ReflectionPad2dFunctor >> elem_cnt >>>>>>>>>>>>>>>>>>>>>>>>>>>%ld !\n", elem_cnt);
    RUN_CUDA_KERNEL((DoCUDAReflectionPad2d<T>), ctx, BlocksNum4ThreadsNum(elem_cnt),
                   x, y, c_idx, h_idx, w_idx, pad_left, pad_top);
  }
};


template<typename T>
struct ReflectionPad2dGradFunctor<DeviceType::kGPU, T> final {
  void operator()(
      DeviceCtx* ctx, const Tensor*  dy, Tensor* dx, int64_t c_idx, int64_t h_idx, int64_t w_idx, int64_t pad_left, int64_t pad_top
    ){
    int64_t  elem_cnt = dy->shape().elem_cnt();
    printf("\n.cu >>>>>>>>>>>>>>>>> ReflectionPad2dGradFunctor >> elem_cnt >>>>>>>>>>>>>>>>>>>>>>>>>>>%ld !\n", elem_cnt);
    RUN_CUDA_KERNEL((DoCUDAReflectionPad2dGrad<T>), ctx, BlocksNum4ThreadsNum(elem_cnt),
                   dy, dx, c_idx, h_idx, w_idx, pad_left, pad_top);
  }
};



OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_REFLECTION_PAD2D_FUNCTOR,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 REFLECTION_PAD2D_DATA_TYPE_GPU_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_REFLECTION_PAD2D_GRAD_FUNCTOR,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 REFLECTION_PAD2D_GRAD_DATA_TYPE_GPU_SEQ);


}  // namespace user_op
}  // namespace oneflow

#endif  // WITH_CUDA