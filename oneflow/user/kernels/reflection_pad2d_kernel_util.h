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
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define REFLECTION_PAD2D_DATA_TYPE_GPU_SEQ \
  REFLECTION_PAD2D_DATA_TYPE_CPU_SEQ   \
  FLOAT16_DATA_TYPE_SEQ

namespace user_op {

template<DeviceType device_type, typename T>
struct ReflectionPad2dFunctor final {
  void operator()(
      DeviceCtx* ctx, const Tensor*  x, Tensor* y, int64_t c_idx, int64_t h_idx, int64_t w_idx, int64_t pad_left, int64_t pad_top
  );
};


template<typename T>
OF_DEVICE_FUNC void DoReflectionPad2d(
    const Tensor*  x, Tensor* y, int64_t c_idx, int64_t h_idx, int64_t w_idx, int64_t pad_left, int64_t pad_top    
) {

  int64_t x_height = x->shape().At(h_idx);
  int64_t x_width = x->shape().At(w_idx);
  int64_t y_height = y->shape().At(h_idx);
  int64_t y_width = y->shape().At(w_idx);

  int64_t ip_x, ip_y;
  T * dest = y->mut_dptr<T>();
  const T* src = x->dptr<T>();

  int64_t elem_cnt = y->shape().At(0);
  int64_t n;
  XPU_1D_KERNEL_LOOP(n, elem_cnt) {
    for(int64_t c = 0; c<y->shape().At(c_idx); c++){
        for(int64_t i = 0; i<y->shape().At(h_idx); i++){
          for(int64_t j = 0; j<y->shape().At(w_idx); j++){
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
            int64_t  dest_index = c * y_width * y_height + i * y_width + j;
            int64_t src_index =  c * x_width * x_height + ip_y * x_width + ip_x;
            //printf("src_index:%ld;  dest_index:%ld\n", src_index, dest_index);
            dest[dest_index] = src[src_index];
            //Memcpy<device_type>(ctx->device_ctx(), y->mut_dptr<T>() + dest_index, x->dptr<T>() + src_index, sizeof_dtype);
          }
        }
    }
  }
}

// macros for functors instantiate(used by reflection_pad2d_kernel_util.cu)
#define INSTANTIATE_REFLECTION_PAD2D_FUNCTOR(device_type_v, dtype_pair)   \
  template struct ReflectionPad2dFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_REFLECTION_PAD2D_KERNEL_UTIL_H_