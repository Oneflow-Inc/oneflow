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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/user/kernels/util_ops_kernel_functor.h"

namespace oneflow {
namespace user_op {
template<typename T>
__global__ void IsNanCudaKernel(bool* y_ptr, const T* x_ptr, const size_t elem_cnt) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { y_ptr[i] = __isnan(x_ptr[i]); }
}

template<typename T>
struct IsNanFunctor<DeviceType::kCUDA, T> {
  void operator()(ep::Stream* stream, bool* y_ptr, const T* x_ptr, const size_t elem_cnt) {
    RUN_CUDA_KERNEL((IsNanCudaKernel<T>), stream, elem_cnt, y_ptr, x_ptr, elem_cnt);
  }
};

template<typename T>
__global__ void IsInfCudaKernel(bool* y_ptr, const T* x_ptr, const size_t elem_cnt) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { y_ptr[i] = __isinf(x_ptr[i]); }
}

template<typename T>
struct IsInfFunctor<DeviceType::kCUDA, T> {
  void operator()(ep::Stream* stream, bool* y_ptr, const T* x_ptr, const size_t elem_cnt) {
    RUN_CUDA_KERNEL((IsInfCudaKernel<T>), stream, elem_cnt, y_ptr, x_ptr, elem_cnt);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_Util_OPS_FUNCTOR, (DeviceType::kCPU),
                                 UTIL_OPS_FUNCTOR_DTYPE_SEQ);

}  // namespace user_op
}  // namespace oneflow
