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
#ifdef WITH_CUDA
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/arange_kernel_util.h"

namespace oneflow {

namespace user_op {

template<typename T>
__global__ void ArangeForwardGpuKernel(const T start, const T delta, const int64_t arange_elem_cnt,
                                       T* out) {
  // Use Loop to set the value
  DoArange<T>(start, delta, arange_elem_cnt, out);
}

template<>
__global__ void ArangeForwardGpuKernel(const half start, const half delta,
                                       const int64_t arange_elem_cnt, half* out) {
  // Use Loop to set the value
  XPU_1D_KERNEL_LOOP(i, arange_elem_cnt) {
    out[i] = start + static_cast<half>(static_cast<float>(i)) * delta;
  }
}

template<typename T>
struct ArangeFunctor<DeviceType::kCUDA, T> final {
  void operator()(ep::Stream* stream, const T start, const T delta, const int64_t arange_elem_cnt,
                  T* out) {
    // The thread num is set as arange_elem_cnt
    RUN_CUDA_KERNEL((ArangeForwardGpuKernel<T>), stream, arange_elem_cnt, start, delta,
                    arange_elem_cnt, out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ARANGE_FUNCTOR, (DeviceType::kCUDA),
                                 ARANGE_DATA_TYPE_SEQ);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ARANGE_FUNCTOR, (DeviceType::kCUDA),
                                 HALF_DATA_TYPE_SEQ);
}  // namespace user_op
}  // namespace oneflow

#endif  // End WITH_CUDA
