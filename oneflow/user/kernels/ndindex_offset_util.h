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
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
namespace oneflow {

#define NDINDEX_OFFSET_DATA_TYPE_SEQ              \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

constexpr int kMaxDims = 6;

template<typename IDX_T>
using Helper = NdIndexOffsetHelper<IDX_T, kMaxDims>;

#ifdef __CUDA_ARCH__
template<typename T>
__forceinline__ __device__ void checkIndexGPU(T index, T dim) {
  if (index > dim) { __trap(); }
}
#endif

template<typename T>
inline void checkIndexCPU(T index, T dim) {
  CHECK_LE(index, dim);
}

#ifdef __CUDA_ARCH__
template<typename T>
__forceinline__ __device__ void checkOffsetGPU(T offset, T dims_elem_cnt) {
  if (offset >= dims_elem_cnt) { __trap(); }
}
#endif

template<typename T>
inline void checkOffsetCPU(T offset, T dims_elem_cnt) {
  CHECK_LE(offset, dims_elem_cnt);
}

namespace user_op {
template<DeviceType device_type, typename T>
struct NdIndexToOffsetFunctor final {
  void operator()(DeviceCtx* ctx, int32_t in_num, int32_t ndim, const T* index,
                  const T* dims_tensor, T* out);
};

template<typename T>
OF_DEVICE_FUNC void DoNdIndexToOffset(int32_t in_num, int32_t ndim, const T* index, const T* dims,
                                      T* out) {
  Helper<T> helper(dims, ndim);
  T offset = helper.NdIndexToOffset(index, in_num);
  // Check the element in `index` is less than `dim`
  for (int i = 0; i < in_num; i++) {
#ifdef __CUDA_ARCH__
    checkIndexGPU(index[i], dims[i]);
#else
    checkIndexCPU(index[i], dims[i]);
#endif
  }
  out[0] = offset;
}

#define INSTANTIATE_NDINDEX_TO_OFFSET_FUNCTOR(device_type_v, dtype_pair) \
  template struct NdIndexToOffsetFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

template<DeviceType device_type, typename T>
struct OffsetToNdIndexFunctor final {
  void operator()(DeviceCtx* ctx, int32_t dims_num, const T* index, const T* dims_tensor, T* out);
};

template<typename T>
OF_DEVICE_FUNC void DoOffsetToNdIndex(int32_t dims_num, const T* offset, const T* dims, T* out) {
  Helper<T> helper(dims, dims_num);
  int offset_val = *offset;
  int dims_elem_cnt = 1;
  for (int i = 0; i < dims_num; i++) { dims_elem_cnt = dims_elem_cnt * dims[i]; }
#ifdef __CUDA_ARCH__
  checkOffsetGPU(offset_val, dims_elem_cnt);
#else
  checkOffsetCPU(offset_val, dims_elem_cnt);
#endif
  helper.OffsetToNdIndex(offset_val, out);
}

#define INSTANTIATE_OFFSET_TO_NDINDEX_FUNCTOR(device_type_v, dtype_pair) \
  template struct OffsetToNdIndexFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

}  // namespace user_op
}  // namespace oneflow
