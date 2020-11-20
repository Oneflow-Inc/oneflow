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
#ifndef ONEFLOW_USER_KERNELS_DIM_GAHTER_SCATTER__UTIL_H_
#define ONEFLOW_USER_KERNELS_DIM_GAHTER_SCATTER__UTIL_H_
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {

#define DIM_GATHER_SCATTER_DATA_TYPE_CPU_SEQ \
  FLOATING_DATA_TYPE_SEQ                     \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

#define DIM_GATHER_SCATTER_DATA_TYPE_GPU_SEQ \
  DIM_GATHER_SCATTER_DATA_TYPE_CPU_SEQ       \
  FLOAT16_DATA_TYPE_SEQ

constexpr int kDimGatherMaxDimCount = 8;

namespace user_op {

template<typename T>
using DimOpIndexNdHelper = NdIndexOffsetHelper<T, kDimGatherMaxDimCount>;

template <typename T>
using BinaryOpFn = void(*)(const T* x, T* y);

template<typename T>
struct DeviceBinOp {
  OF_DEVICE_FUNC static void Add(const T* x, T* y) {
#ifdef __CUDA_ARCH__
    gpu_atomic_add(y, *x);  // TODO:(YaoChi), refine add using float16 -> half -> float -> half
#else
    *y += *x;
#endif
  }

  OF_DEVICE_FUNC static void Update(const T* x, T* y) {
    *y = *x;
  }
};


}  // namespace user_op
}  // namespace oneflow

#endif