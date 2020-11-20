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
#include "oneflow/core/kernel/util/cuda_kernel_util.h"
#include "oneflow/user/kernels/dim_scatter_kernel_util.h"

namespace oneflow {
namespace user_op {

IMPLEMENT_DIMSCATTER_GPUFUNCTOR(Add);
IMPLEMENT_DIMSCATTER_GPUFUNCTOR(Update);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_DIM_SCATTER_ADD_FUNCTOR, (DeviceType::kGPU),
                                 DIM_GATHER_SCATTER_DATA_TYPE_GPU_SEQ, INDEX_DATA_TYPE_SEQ);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_DIM_SCATTER_UPDATE_FUNCTOR, (DeviceType::kGPU),
                                 DIM_GATHER_SCATTER_DATA_TYPE_GPU_SEQ, INDEX_DATA_TYPE_SEQ);
}  // namespace user_op
}  // namespace oneflow

#endif  // WITH_CUDA
