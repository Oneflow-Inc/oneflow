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
#ifndef ONEFLOW_CORE_NDARRAY_XPU_UTIL_H_
#define ONEFLOW_CORE_NDARRAY_XPU_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

#if defined(__CUDACC__)
#define XPU_1D_KERNEL_LOOP_BEGIN(i, n) CUDA_1D_KERNEL_LOOP(i, n) {
#define XPU_1D_KERNEL_LOOP_END() }
#else
#define XPU_1D_KERNEL_LOOP_BEGIN(i, n) MultiThreadLoop(n, [&](size_t i) {
#define XPU_1D_KERNEL_LOOP_END() \
  });
#endif

#if defined(__CUDACC__)
#define XPU_1D_KERNEL_LOOP(i, n) CUDA_1D_KERNEL_LOOP(i, n)
#else
#define XPU_1D_KERNEL_LOOP(i, n) FOR_RANGE(int64_t, i, 0, n)
#endif

#if defined(__CUDACC__)
#define XPU_BLOAD_THREAD_2D_KERNEL_LOOP(i, j, m, n)     \
  for (int64_t i = blockIdx.x; i < (m); i += gridDim.x) \
    for (int64_t j = threadIdx.x; j < (n); j += blockDim.x)
#else
#define XPU_BLOAD_THREAD_2D_KERNEL_LOOP(i, j, m, n) \
  for (int64_t i = 0; i < (m); ++i)                 \
    for (int64_t j = 0; j < (n); ++j)
#endif

#if defined(__CUDACC__)
#define OF_GLOBAL_FUNC __global__
#else
#define OF_GLOBAL_FUNC
#endif

#define GET_SEQ(n) OF_PP_CAT(OF_PP_CAT(GET_SEQ_, n), )
#define GET_SEQ_0 OF_PP_MAKE_TUPLE_SEQ(0)
#define GET_SEQ_1 GET_SEQ_0 OF_PP_MAKE_TUPLE_SEQ(1)
#define GET_SEQ_2 GET_SEQ_1 OF_PP_MAKE_TUPLE_SEQ(2)
#define GET_SEQ_3 GET_SEQ_2 OF_PP_MAKE_TUPLE_SEQ(3)
#define GET_SEQ_4 GET_SEQ_3 OF_PP_MAKE_TUPLE_SEQ(4)
#define GET_SEQ_5 GET_SEQ_4 OF_PP_MAKE_TUPLE_SEQ(5)
#define GET_SEQ_6 GET_SEQ_5 OF_PP_MAKE_TUPLE_SEQ(6)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_UTIL_H_
