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
#include "oneflow/user/kernels/where_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename CondT>
__global__ void CudaWhere(const int64_t elem_cnt, const CondT* cond, const T* lhs, const T* rhs,
                          T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { out[i] = static_cast<bool>(cond[i]) ? lhs[i] : rhs[i]; }
}

}  // namespace

template<typename T, typename CondT>
struct WhereKernelUtil<DeviceType::kGPU, T, CondT> {
  static void Where(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                    const T* rhs, T* out) {
    RUN_CUDA_KERNEL((CudaWhere<T, CondT>), ctx, elem_cnt, elem_cnt, cond, lhs, rhs, out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_WHERE_FUNCTOR, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
