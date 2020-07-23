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
#ifndef ONEFLOW_CORE_KERNEL_NONZERO_KERNEL_CU_H_
#define ONEFLOW_CORE_KERNEL_NONZERO_KERNEL_CU_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

template<typename T, typename OutputIter>
cudaError_t CubSelectFlagged(cudaStream_t stream, int num_items, void* tmp, size_t& tmp_bytes,
                             const T* flags, OutputIter out, int32_t* num_selected);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NONZERO_KERNEL_CU_H_
