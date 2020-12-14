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
#ifndef ONEFLOW_USER_KERNELS_REFLECTION_PAD2D_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_REFLECTION_PAD2D_KERNEL_UTIL_H_
#ifdef WITH_CUDA
#include "oneflow/core/kernel/util/cuda_kernel_util.h"
#endif  // WITH_CUDA
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {
  
#define REFLECTION_PAD2D_DATA_TYPE_CPU_SEQ \
  FLOATING_DATA_TYPE_SEQ                     \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, DataType::kInt8) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define REFLECTION_PAD2D_DATA_TYPE_GPU_SEQ \
  REFLECTION_PAD2D_DATA_TYPE_CPU_SEQ   \
  FLOAT16_DATA_TYPE_SEQ
  

#define REFLECTION_PAD2D_GRAD_DATA_TYPE_CPU_SEQ \
  FLOATING_DATA_TYPE_SEQ                     \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, DataType::kInt8) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define REFLECTION_PAD2D_GRAD_DATA_TYPE_GPU_SEQ \
  REFLECTION_PAD2D_GRAD_DATA_TYPE_CPU_SEQ   \
  FLOAT16_DATA_TYPE_SEQ

namespace user_op {


template<typename T>
struct DeviceAdd {
  OF_DEVICE_FUNC static void Invoke(const T* x, T* y) {
#ifdef __CUDA_ARCH__
    gpu_atomic_add(y, *x);  // TODO:(ZhaoLuyang), refine add using float16 -> half -> float -> half
#else
    *y += *x;
#endif
  };
};


template<DeviceType device_type, typename T>
struct ReflectionPad2dFunctor final {
  void operator()(
      DeviceCtx* ctx, const T* src, T * dest,
      int64_t n_batch, int64_t n_channel,int64_t y_height, int64_t y_width,
      int64_t x_height, int64_t x_width, int64_t pad_left, int64_t pad_top
  );
};


template<DeviceType device_type, typename T>
struct ReflectionPad2dGradFunctor final {
  void operator()(
      DeviceCtx* ctx, const T* src, T * dest,
      int64_t n_batch, int64_t n_channel,int64_t dy_height, int64_t dy_width,
      int64_t dx_height, int64_t dx_width, int64_t pad_left, int64_t pad_top
  );
};


template<typename T>
OF_DEVICE_FUNC void DoReflectionPad2d(
    const T* src, T * dest,
    int64_t n_batch, int64_t n_channel,int64_t y_height, int64_t y_width,
    int64_t x_height, int64_t x_width, int64_t pad_left, int64_t pad_top
) {
  int64_t dest_num = n_channel * y_height * y_width;
  int64_t src_num = n_channel * x_height * x_width;
  XPU_1D_KERNEL_LOOP(n, n_batch){
    int64_t ip_x, ip_y;
    for(int64_t c = 0; c<n_channel; c++){
      for(int64_t i = 0; i<y_height; i++){
        for(int64_t j = 0; j<y_width; j++){
          if(j < pad_left){
            ip_x = pad_left * 2 - j;
          }else if( j >= pad_left && j < x_width + pad_left){
            ip_x = j;
          }else{
            ip_x = (x_width + pad_left - 1) * 2 - j;
          }

          if(i<pad_top){
            ip_y = pad_top * 2 - i;
          }else if(i >= pad_top && i < x_height + pad_top){
            ip_y = i;
          }else{
            ip_y = (x_height + pad_top - 1) * 2 - i;
          }
          ip_x = ip_x - pad_left;
          ip_y = ip_y - pad_top;
          int64_t num = n * dest_num + c*n_channel + i*y_height + j;
          int64_t  dest_index = n * dest_num + c * y_width * y_height + i * y_width + j;
          int64_t src_index = n * src_num + c * x_width * x_height + ip_y * x_width + ip_x;
          dest[dest_index] = src[src_index];
        }
      }
    }
  }
  
}


template<typename T>
OF_DEVICE_FUNC void DoReflectionPad2dGrad(
    const T* src, T * dest, int64_t n_batch, int64_t n_channel,
    int64_t dy_height, int64_t dy_width, int64_t dx_height, int64_t dx_width, 
    int64_t pad_left, int64_t pad_top 
) {
  int64_t src_num = n_channel * dy_height * dy_width;
  int64_t dest_num = n_channel * dx_height * dx_width;
  XPU_1D_KERNEL_LOOP(n, n_batch){
    int64_t ip_x, ip_y;
    for(int64_t c = 0; c<n_channel; c++){
      for(int64_t i = 0; i<dy_height; i++){
        for(int64_t j = 0; j<dy_width; j++){
          if(j < pad_left){
            ip_x = pad_left * 2 - j;
          }else if( j >= pad_left && j < dx_width + pad_left){
            ip_x = j;
          }else{
            ip_x = (dx_width + pad_left - 1) * 2 - j;
          }
        
          if(i<pad_top){
            ip_y = pad_top * 2 - i;
          }else if(i >= pad_top && i < dx_height + pad_top){
            ip_y = i;
          }else{
            ip_y = (dx_height + pad_top - 1) * 2 - i;
          }
          ip_x = ip_x - pad_left;
          ip_y = ip_y - pad_top;
          int64_t src_index = n * src_num + c * dy_width * dy_height + i * dy_width + j;
          int64_t dest_index = n * dest_num + c * dx_width * dx_height + ip_y * dx_width + ip_x;
          //DeviceAdd<T>::Invoke(src + src_index, dest + dest_index);
          dest[dest_index] += src[src_index];
        }
      }
    }
  }
}


// macros for functors instantiate(used by reflection_pad2d_kernel_util.cu)
#define INSTANTIATE_REFLECTION_PAD2D_FUNCTOR(device_type_v, dtype_pair)   \
  template struct ReflectionPad2dFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

#define INSTANTIATE_REFLECTION_PAD2D_GRAD_FUNCTOR(device_type_v, dtype_pair)   \
  template struct ReflectionPad2dGradFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_REFLECTION_PAD2D_KERNEL_UTIL_H_