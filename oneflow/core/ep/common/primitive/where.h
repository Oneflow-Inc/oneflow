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

#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/ep/common/primitive/util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {
namespace ep {
namespace primitive {

namespace where_impl {

constexpr size_t kMaxNumDims = 8;

template<typename T>
struct WhereFunctor {
  OF_DEVICE_FUNC WhereFunctor() {}

  OF_DEVICE_FUNC T operator()(bool cond, T x, T y) const { return cond ? x : y; }
};

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

inline bool IsDimsEquals(size_t ndim, const int64_t* a_dims, const int64_t* b_dims) {
  for (size_t i = 0; i < ndim; ++i) {
    if (a_dims[i] != b_dims[i]) { return false; }
  }
  return true;
}

inline void GetCompactBroadcastDims(const size_t cond_ndim, const int64_t* cond_dims,
                                    const size_t x_ndim, const int64_t* x_dims, const size_t y_ndim,
                                    const int64_t* y_dims, size_t& compt_ndim,
                                    int64_t* cmpt_cond_dims, int64_t* cmpt_x_dims,
                                    int64_t* cmpt_y_dims, int64_t* cmpt_z_dims) {
  size_t max_ndim = std::max(x_ndim, y_ndim);
  max_ndim = std::max(max_ndim, cond_ndim);
  CHECK_LE(max_ndim, kMaxNumDims);

  auto MakeGetDimSize = [max_ndim](size_t ndim, const int64_t* dims) {
    size_t lpad = max_ndim - ndim;
    return [lpad, dims](int dim) -> int64_t { return dim < lpad ? 1 : dims[dim - lpad]; };
  };
  auto GetCondDimSize = MakeGetDimSize(cond_ndim, cond_dims);
  auto GetXDimSize = MakeGetDimSize(x_ndim, x_dims);
  auto GetYDimSize = MakeGetDimSize(y_ndim, y_dims);

  compt_ndim = 0;
  bool cond_pred_dim_broadcast = false;
  bool x_pred_dim_broadcast = false;
  bool y_pred_dim_broadcast = false;
  for (int i = 0; i < max_ndim; ++i) {
    int64_t cond_dim_size = GetCondDimSize(i);
    int64_t x_dim_size = GetXDimSize(i);
    int64_t y_dim_size = GetYDimSize(i);
    int64_t dim_size = std::max(std::max(x_dim_size, y_dim_size), cond_dim_size);
    if (dim_size == 1) { continue; }
    bool cond_broadcast = (cond_dim_size == 1);
    bool x_broadcast = (x_dim_size == 1);
    bool y_broadcast = (y_dim_size == 1);
    if (compt_ndim > 0 && cond_broadcast == cond_pred_dim_broadcast
        && x_broadcast == x_pred_dim_broadcast && y_broadcast == y_pred_dim_broadcast) {
      cmpt_cond_dims[compt_ndim - 1] *= cond_dim_size;
      cmpt_x_dims[compt_ndim - 1] *= x_dim_size;
      cmpt_y_dims[compt_ndim - 1] *= y_dim_size;
      cmpt_z_dims[compt_ndim - 1] *= dim_size;
    } else {
      cmpt_cond_dims[compt_ndim] = cond_dim_size;
      cmpt_x_dims[compt_ndim] = x_dim_size;
      cmpt_y_dims[compt_ndim] = y_dim_size;
      cmpt_z_dims[compt_ndim] = dim_size;
      compt_ndim += 1;
      cond_pred_dim_broadcast = cond_broadcast;
      x_pred_dim_broadcast = x_broadcast;
      y_pred_dim_broadcast = y_broadcast;
    }
  }
}

template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  OF_DEVICE_FUNC Pack() {}

  PackType<T, N> storage;
  T elem[N];
};

template<typename IndexType, size_t ndim, size_t cond_type_size, size_t data_type_size,
         size_t cond_pack_size, size_t x_pack_size, size_t y_pack_size>
void LaunchKernel(Stream* stream, const int64_t* cond_dims, const void* cond, const int64_t* x_dims,
                  const void* x, const int64_t* y_dims, const void* y, const int64_t* z_dims,
                  void* z);

template<size_t ndim, size_t cond_type_size, size_t data_type_size, size_t cond_pack_size,
         size_t x_pack_size, size_t y_pack_size>
void LaunchByDispatchIndexType(Stream* stream, const int64_t* cond_dims, const void* cond,
                               const int64_t* x_dims, const void* x, const int64_t* y_dims,
                               const void* y, const int64_t* z_dims, void* z) {
  const size_t elem_cnt = GetElementCount(ndim, z_dims);
  if (elem_cnt < GetMaxVal<int32_t>()) {
    return LaunchKernel<int32_t, ndim, cond_type_size, data_type_size, cond_pack_size, x_pack_size,
                        y_pack_size>(stream, cond_dims, cond, x_dims, x, y_dims, y, z_dims, z);
  } else {
    return LaunchKernel<int64_t, ndim, cond_type_size, data_type_size, cond_pack_size, x_pack_size,
                        y_pack_size>(stream, cond_dims, cond, x_dims, x, y_dims, y, z_dims, z);
  }
}

template<size_t ndim, size_t cond_type_size, size_t data_type_size, size_t max_pack_size>
size_t GetPackSize(const int64_t* cond_dims, const void* cond, const int64_t* x_dims, const void* x,
                   const int64_t* y_dims, const void* y, const int64_t* z_dims, const void* z) {
  static_assert(max_pack_size > 0 && (max_pack_size & (max_pack_size - 1)) == 0, "");
  using T = typename std::aligned_storage<data_type_size, data_type_size>::type;
  using CondT = typename std::aligned_storage<cond_type_size, cond_type_size>::type;
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

template<size_t ndim, size_t cond_type_size, size_t data_type_size>
void LaunchByDispatchPackSize(Stream* stream, int64_t* cond_dims, const void* cond, int64_t* x_dims,
                              const void* x, int64_t* y_dims, const void* y, int64_t* z_dims,
                              void* z) {
  constexpr size_t kMaxPackSize = 4;
  size_t pack_size = GetPackSize<ndim, cond_type_size, data_type_size, kMaxPackSize>(
      cond_dims, cond, x_dims, x, y_dims, y, z_dims, z);
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

  void (*func)(Stream* /*stream*/, const int64_t* /*cond_dims*/, const void* /*cond*/,
               const int64_t* /*x_dims*/, const void* /*x*/, const int64_t* /*y_dims*/,
               const void* /*y*/, const int64_t* /*z_dims*/, void* /*z*/) = nullptr;

#define IF(c, x, y)                                                                  \
  if (cond_pack_size == c && x_pack_size == x && y_pack_size == y) {                 \
    func = LaunchByDispatchIndexType<ndim, cond_type_size, data_type_size, c, x, y>; \
  }
#define ELIF(c, x, y) else IF(c, x, y)
#define ELSE         \
  else {             \
    UNIMPLEMENTED(); \
  }

  if (pack_size == 1) {
    func = LaunchByDispatchIndexType<ndim, cond_type_size, data_type_size, 1, 1, 1>;
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

#undef IF
#undef ELIF
#undef ELSE

  func(stream, cond_dims, cond, x_dims, x, y_dims, y, z_dims, z);
}

template<size_t ndim>
void LaunchByDispatchTypeSize(Stream* stream, DataType cond_type, DataType data_type,
                              int64_t* cond_dims, const void* cond, int64_t* x_dims, const void* x,
                              int64_t* y_dims, const void* y, int64_t* z_dims, void* z) {
  size_t cond_type_size = GetSizeOfDataType(cond_type);
  size_t data_type_size = GetSizeOfDataType(data_type);

#define IF(c, d)                                                                                \
  if (cond_type_size == c && data_type_size == d) {                                             \
    LaunchByDispatchPackSize<ndim, c, d>(stream, cond_dims, cond, x_dims, x, y_dims, y, z_dims, \
                                         z);                                                    \
  }
#define ELIF(c, d) else IF(c, d)
#define ELSE         \
  else {             \
    UNIMPLEMENTED(); \
  }

  IF(1, 1)
  ELIF(1, 2)
  ELIF(1, 4)
  ELIF(1, 8)
  ELIF(4, 1)
  ELIF(4, 2)
  ELIF(4, 4)
  ELIF(4, 8)
  ELSE

#undef IF
#undef ELIF
#undef ELSE
}

void LaunchByDispatchNDim(Stream* stream, DataType cond_type, DataType data_type, size_t ndim,
                          int64_t* cond_dims, const void* cond, int64_t* x_dims, const void* x,
                          int64_t* y_dims, const void* y, int64_t* z_dims, void* z) {
#define IF(n)                                                                                     \
  if (ndim == n) {                                                                                \
    LaunchByDispatchTypeSize<n>(stream, cond_type, data_type, cond_dims, cond, x_dims, x, y_dims, \
                                y, z_dims, z);                                                    \
  }
#define ELIF(n) else IF(n)
#define ELSE         \
  else {             \
    UNIMPLEMENTED(); \
  }

  CHECK_GT(ndim, 0);
  IF(1)
  ELIF(2)
  ELIF(3)
  ELIF(4)
  ELSE

#undef IF
#undef ELIF
#undef ELSE
}

}  // namespace where_impl

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_COMMON_PRIMITIVE_WHERE_H_
