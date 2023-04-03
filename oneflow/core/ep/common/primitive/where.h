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
#ifndef ONEFLOW_CORE_EP_COMMON_PRIMITIVE_WHERE_H_
#define ONEFLOW_CORE_EP_COMMON_PRIMITIVE_WHERE_H_

#include "oneflow/core/ep/include/primitive/where.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/ep/common/primitive/util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {
namespace ep {
namespace primitive {

namespace {

constexpr size_t kMaxNumDims = 8;

template<typename R, typename Cond, typename X, typename Y>
struct WhereElemwiseFunctor {
  OF_DEVICE_FUNC WhereElemwiseFunctor() {}

  OF_DEVICE_FUNC R operator()(Cond cond, X x, Y y) const { return cond ? x : y; }
};

template<typename T, typename CondT>
using WhereFunctor = WhereElemwiseFunctor<T, CondT, T, T>;

template<size_t NDIM, typename IndexType>
struct BroadcastElementwiseWhereParams {
  NdIndexOffsetHelper<IndexType, NDIM> cond_index_helper;
  NdIndexOffsetHelper<IndexType, NDIM> x_index_helper;
  NdIndexOffsetHelper<IndexType, NDIM> y_index_helper;
  NdIndexOffsetHelper<IndexType, NDIM> z_index_helper;
  IndexType cond_index_mask[NDIM];
  IndexType x_index_mask[NDIM];
  IndexType y_index_mask[NDIM];
  IndexType elem_cnt{};
  const void* cond{};
  const void* x{};
  const void* y{};
  void* z{};
};

template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed {
  OF_DEVICE_FUNC Packed() {
    // do nothing
  }
  union {
    T elem[pack_size];
  };
};

inline bool IsDimsEquals(size_t ndim, const int64_t* a_dims, const int64_t* b_dims) {
  for (size_t i = 0; i < ndim; ++i) {
    if (a_dims[i] != b_dims[i]) { return false; }
  }
  return true;
}

// Calculate compact broadcast dimensions
// For example:
//   [1, 2, 8] and [4, 2, 8] can be compacted to [1, 16] and [4, 16]
//   [4, 1, 8] and [8] -> [4, 8] and [1, 8]
//   [1, 1, 8] and [4, 2, 8] -> [1, 8] and [8, 8]
// after compacting, cond, x, y will have the same number of dims,
// z_dims is the broadcast dims of compacted cond, x, y dims.
inline void GetCompactBroadcastDims(const size_t num_cond_ndims, const int64_t* cond_dims,
                                    const size_t num_x_dims, const int64_t* x_dims,
                                    const size_t num_y_dims, const int64_t* y_dims,
                                    size_t* compact_num_dims, int64_t* compact_cond_dims,
                                    int64_t* compact_x_dims, int64_t* compact_y_dims,
                                    int64_t* compact_z_dims) {
  size_t max_num_dims = std::max(std::max(num_x_dims, num_y_dims), num_cond_ndims);
  CHECK_LE(max_num_dims, kMaxNumDims);

  auto MakeGetDimSize = [max_num_dims](size_t ndim, const int64_t* dims) {
    size_t lpad = max_num_dims - ndim;
    return [lpad, dims](int dim) -> int64_t { return dim < lpad ? 1 : dims[dim - lpad]; };
  };
  auto GetCondDimSize = MakeGetDimSize(num_cond_ndims, cond_dims);
  auto GetXDimSize = MakeGetDimSize(num_x_dims, x_dims);
  auto GetYDimSize = MakeGetDimSize(num_y_dims, y_dims);

  size_t& num_dims = *compact_num_dims;
  num_dims = 0;
  bool cond_pred_dim_broadcast = false;
  bool x_pred_dim_broadcast = false;
  bool y_pred_dim_broadcast = false;
  for (int i = 0; i < max_num_dims; ++i) {
    int64_t cond_dim_size = GetCondDimSize(i);
    int64_t x_dim_size = GetXDimSize(i);
    int64_t y_dim_size = GetYDimSize(i);
    int64_t dim_size = std::max(std::max(x_dim_size, y_dim_size), cond_dim_size);
    if (dim_size == 1) { continue; }
    bool cond_broadcast = (cond_dim_size == 1);
    bool x_broadcast = (x_dim_size == 1);
    bool y_broadcast = (y_dim_size == 1);
    if (*compact_num_dims > 0 && cond_broadcast == cond_pred_dim_broadcast
        && x_broadcast == x_pred_dim_broadcast && y_broadcast == y_pred_dim_broadcast) {
      compact_cond_dims[num_dims - 1] *= cond_dim_size;
      compact_x_dims[num_dims - 1] *= x_dim_size;
      compact_y_dims[num_dims - 1] *= y_dim_size;
      compact_z_dims[num_dims - 1] *= dim_size;
    } else {
      compact_cond_dims[num_dims] = cond_dim_size;
      compact_x_dims[num_dims] = x_dim_size;
      compact_y_dims[num_dims] = y_dim_size;
      compact_z_dims[num_dims] = dim_size;
      num_dims += 1;
      cond_pred_dim_broadcast = cond_broadcast;
      x_pred_dim_broadcast = x_broadcast;
      y_pred_dim_broadcast = y_broadcast;
    }
  }
}

template<typename T, typename CondT, typename IndexT, size_t ndim, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
void LaunchKernel(Stream* stream, const int64_t* cond_dims, const int64_t* x_dims,
                  const int64_t* y_dims, const int64_t* z_dims, const CondT* cond, const T* x,
                  const T* y, T* z);

template<typename T, typename CondT>
void LaunchScalarKernel(Stream* stream, const CondT* cond, const T* x, const T* y, T* z);

template<typename T, typename CondT, size_t ndim, size_t cond_pack_size, size_t x_pack_size,
         size_t y_pack_size>
void LaunchByDispatchIndexType(Stream* stream, int64_t* cond_dims, int64_t* x_dims, int64_t* y_dims,
                               int64_t* z_dims, const CondT* cond, const T* x, const T* y, T* z) {
  const size_t elem_cnt = GetElementCount(ndim, z_dims);
  if (elem_cnt < GetMaxVal<int32_t>()) {
    return LaunchKernel<T, CondT, int32_t, ndim, cond_pack_size, x_pack_size, y_pack_size>(
        stream, cond_dims, x_dims, y_dims, z_dims, cond, x, y, z);
  } else {
    return LaunchKernel<T, CondT, int64_t, ndim, cond_pack_size, x_pack_size, y_pack_size>(
        stream, cond_dims, x_dims, y_dims, z_dims, cond, x, y, z);
  }
}

template<typename T, typename CondT, size_t ndim, size_t max_pack_size>
size_t GetPackSize(const int64_t* cond_dims, const int64_t* x_dims, const int64_t* y_dims,
                   const int64_t* z_dims, const CondT* cond, const T* x, const T* y, const T* z) {
  static_assert(max_pack_size > 0 && (max_pack_size & (max_pack_size - 1)) == 0, "");
  CHECK_GT(z_dims[ndim - 1], 1);
  for (size_t pack_size = max_pack_size; pack_size >= 2; pack_size /= 2) {
    if (!IsPackSizeSupported<T>(pack_size, ndim, z_dims, z)) { continue; }
    if (x_dims[ndim - 1] != 1 && !IsPackSizeSupported<T>(pack_size, ndim, x_dims, x)) { continue; }
    if (y_dims[ndim - 1] != 1 && !IsPackSizeSupported<T>(pack_size, ndim, y_dims, y)) { continue; }
    if (cond_dims[ndim - 1] != 1 && !IsPackSizeSupported<CondT>(pack_size, ndim, cond_dims, cond)) {
      continue;
    }
    return pack_size;
  }
  return 1;
}

template<typename T, typename CondT, size_t ndim>
void LaunchByDispatchPackSize(Stream* stream, int64_t* cond_dims, int64_t* x_dims, int64_t* y_dims,
                              int64_t* z_dims, const CondT* cond, const T* x, const T* y, T* z) {
  static_assert(ndim > 0, "");
  constexpr size_t kMaxPackSize = 4;
  size_t pack_size =
      GetPackSize<T, CondT, ndim, kMaxPackSize>(cond_dims, x_dims, y_dims, z_dims, cond, x, y, z);
  size_t cond_pack_size = 1;
  size_t x_pack_size = 1;
  size_t y_pack_size = 1;
  if (pack_size > 1) {
    if (cond_dims[ndim - 1] != 1) {
      cond_dims[ndim - 1] /= pack_size;
      cond_pack_size = pack_size;
    }
    if (x_dims[ndim - 1] != 1) {
      x_dims[ndim - 1] /= pack_size;
      x_pack_size = pack_size;
    }
    if (y_dims[ndim - 1] != 1) {
      y_dims[ndim - 1] /= pack_size;
      y_pack_size = pack_size;
    }
    z_dims[ndim - 1] /= pack_size;
  }

#define IF(cp, xp, yp)                                                                       \
  if (cond_pack_size == cp && x_pack_size == xp && y_pack_size == yp) {                      \
    LaunchByDispatchIndexType<T, CondT, ndim, cp, xp, yp>(stream, cond_dims, x_dims, y_dims, \
                                                          z_dims, cond, x, y, z);            \
  }
#define ELIF(cp, xp, yp) else IF(cp, xp, yp)
#define ELSE         \
  else {             \
    UNIMPLEMENTED(); \
  }

  if (pack_size == 1) {
    IF(1, 1, 1)
    ELSE
  } else if (pack_size == 2) {
    IF(2, 2, 2)
    ELIF(1, 2, 2)
    ELIF(1, 2, 1)
    ELIF(1, 1, 2)
    ELIF(2, 1, 2)
    ELIF(2, 1, 1)
    ELIF(2, 2, 1)
    ELSE
  } else if (pack_size == 4) {
    IF(4, 4, 4)
    ELIF(1, 4, 4)
    ELIF(1, 4, 1)
    ELIF(1, 1, 4)
    ELIF(4, 1, 4)
    ELIF(4, 1, 1)
    ELIF(4, 4, 1)
    ELSE
  }
  ELSE

#undef IF
#undef ELIF
#undef ELSE
}

template<typename T, typename CondT>
void LaunchByDispatchNDim(Stream* stream, size_t ndim, int64_t* cond_dims, int64_t* x_dims,
                          int64_t* y_dims, int64_t* z_dims, const CondT* cond, const T* x,
                          const T* y, T* z) {
#define ELIF(n)                                                                                  \
  else if (ndim == n) {                                                                          \
    LaunchByDispatchPackSize<T, CondT, n>(stream, cond_dims, x_dims, y_dims, z_dims, cond, x, y, \
                                          z);                                                    \
  }
#define ELSE         \
  else {             \
    UNIMPLEMENTED(); \
  }

  if (ndim == 0) { LaunchScalarKernel<T, CondT>(stream, cond, x, y, z); }
  ELIF(1)
  ELIF(2)
  ELIF(3)
  ELIF(4)
  ELSE

#undef IF
#undef ELIF
#undef ELSE
}

template<template<typename, typename> class Prim>
std::unique_ptr<Where> NewWhere(DataType cond_type, DataType data_type, size_t max_num_dims) {
  if (max_num_dims > kMaxNumDims) { return nullptr; }

  const size_t data_type_size = GetSizeOfDataType(data_type);

#define IF(ctype, dtype_size)                                              \
  if (cond_type == ctype && data_type_size == dtype_size) {                \
    using T = typename std::aligned_storage<dtype_size, dtype_size>::type; \
    using CondT = DataTypeToType<ctype>;                                   \
    return std::unique_ptr<Where>(new Prim<T, CondT>());                   \
  }
#define ELIF(ctype, dtype_size) else IF(ctype, dtype_size)
#define ELSE        \
  else {            \
    return nullptr; \
  }

  IF(DataType::kBool, 1)
  ELIF(DataType::kBool, 2)
  ELIF(DataType::kBool, 4)
  ELIF(DataType::kBool, 8)
  ELIF(DataType::kInt32, 1)
  ELIF(DataType::kInt32, 2)
  ELIF(DataType::kInt32, 4)
  ELIF(DataType::kInt32, 8)
  ELIF(DataType::kInt64, 1)
  ELIF(DataType::kInt64, 2)
  ELIF(DataType::kInt64, 4)
  ELIF(DataType::kInt64, 8)
  ELSE

#undef IF
#undef ELIF
#undef ELSE
}

}  // namespace

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_COMMON_PRIMITIVE_WHERE_H_
