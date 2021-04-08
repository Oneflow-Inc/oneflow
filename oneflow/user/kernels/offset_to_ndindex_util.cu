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
#include "oneflow/user/kernels/ndindex_offset_util.h"

namespace oneflow {

namespace user_op {

template<typename T>
__global__ void OffsetToNdIndexForwardGpuKernel(int32_t dims_num, const T* offset, const T* dims,
                                                T* out) {
  DoOffsetToNdIndex<T>(dims_num, offset, dims, out);
}

template<typename T>
struct OffsetToNdIndexFunctor<DeviceType::kGPU, T> final {
  void operator()(DeviceCtx* ctx, int32_t dims_num, const T* offset, const T* dims, T* out) {
    RUN_CUDA_KERNEL((OffsetToNdIndexForwardGpuKernel<T>), ctx, dims_num, dims_num, offset, dims,
                    out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_OFFSET_TO_NDINDEX_FUNCTOR, (DeviceType::kGPU),
                                 NDINDEX_OFFSET_DATA_TYPE_SEQ);
}  // namespace user_op
}  // namespace oneflow

#endif  // End WITH_CUDA
