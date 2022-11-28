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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/ndarray/ndarray_reduce_impl.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

#define SPECIALIZE_CPU_NDARRAY_REDUCE_IMPL(struct_name)                                        \
  template<typename T, template<typename> class binary_func>                                   \
  struct struct_name<DeviceType::kCPU, T, binary_func> final {                                 \
    using RetT = typename BinaryFuncTrait<binary_func, T>::return_type;                        \
    static bool Matched(const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& x) {       \
      return false;                                                                            \
    }                                                                                          \
    static void Reduce(ep::Stream* stream, const XpuVarNdarray<RetT>& y,                       \
                       const XpuVarNdarray<const T>& x, const XpuVarNdarray<T>& tmp_storage) { \
      UNIMPLEMENTED();                                                                         \
    }                                                                                          \
  }
SPECIALIZE_CPU_NDARRAY_REDUCE_IMPL(NdarrayScalarReduce);
SPECIALIZE_CPU_NDARRAY_REDUCE_IMPL(NdarrayMatrixRowReduce);
SPECIALIZE_CPU_NDARRAY_REDUCE_IMPL(NdarrayMatrixColReduce);
SPECIALIZE_CPU_NDARRAY_REDUCE_IMPL(NdarrayXYZCubeXZReduce);
#undef SPECIALIZE_CPU_NDARRAY_REDUCE_IMPL

#define INSTANTIATE_NDARRAY_REDUCE_IMPL(dtype, binary_func)                                       \
  template struct NdarrayScalarReduce<DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype), binary_func>;    \
  template struct NdarrayMatrixRowReduce<DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype), binary_func>; \
  template struct NdarrayMatrixColReduce<DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype), binary_func>; \
  template struct NdarrayXYZCubeXZReduce<DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype), binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_IMPL,
                                 ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ
                                     UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 REDUCE_BINARY_FUNC_SEQ);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_IMPL, FLOATING_DATA_TYPE_SEQ,
                                 NANSUM_REDUCE_BINARY_FUNC_SEQ);

template<typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayReduceCoreWrapper<DeviceType::kCPU, T, NDIMS, binary_func> final {
  static void ReduceAxis(ep::Stream* stream, const XpuReducedNdarray<T, NDIMS>& dst_reduced,
                         const XpuReducedNdarray<T, NDIMS>& x, int axis) {
    NdarrayReduceCore<T, NDIMS, binary_func>::ReduceAxis(dst_reduced, x, axis);
  }
};

#define INSTANTIATE_NDARRAY_REDUCE_CORE_WRAPPER(dtype_pair, NDIMS, binary_func)                   \
  template struct NdarrayReduceCoreWrapper<DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS, \
                                           binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_CORE_WRAPPER,
                                 ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ
                                     UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 DIM_SEQ, REDUCE_BINARY_FUNC_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_CORE_WRAPPER, FLOATING_DATA_TYPE_SEQ,
                                 DIM_SEQ, NANSUM_REDUCE_BINARY_FUNC_SEQ);

}  // namespace oneflow
