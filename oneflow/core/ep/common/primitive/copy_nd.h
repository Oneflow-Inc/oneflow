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

#ifndef ONEFLOW_CORE_PRIMITIVE_COMMON_COPY_ND_H_
#define ONEFLOW_CORE_PRIMITIVE_COMMON_COPY_ND_H_

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<size_t num_dims, typename IndexType>
struct CopyNdKernelParams {
  NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> dst_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> copy_index_helper;
  IndexType dst_pos[num_dims];
  IndexType src_pos[num_dims];
  IndexType count{};
  const void* src{};
  void* dst{};
};

template<size_t max_movement_size>
size_t GetMovementSize(size_t elem_size, size_t num_dims, void* dst, const int64_t* dst_dims,
                       const int64_t* dst_pos, const void* src, const int64_t* src_dims,
                       const int64_t* src_pos, const int64_t* extent) {
  static_assert(max_movement_size > 0 && (max_movement_size & (max_movement_size - 1)) == 0, "");
  CHECK_GT(elem_size, 0);
  CHECK_EQ((elem_size & (elem_size - 1)), 0);
  CHECK_EQ(max_movement_size % elem_size, 0);
  const int64_t last_dst_dim_size = dst_dims[num_dims - 1] * elem_size;
  const int64_t last_dst_pos = dst_pos[num_dims - 1] * elem_size;
  const int64_t last_src_dim_size = src_dims[num_dims - 1] * elem_size;
  const int64_t last_src_pos = src_pos[num_dims - 1] * elem_size;
  const int64_t last_extent = extent[num_dims - 1] * elem_size;
  auto src_ptr = reinterpret_cast<std::uintptr_t>(src);
  auto dst_ptr = reinterpret_cast<std::uintptr_t>(dst);
  for (size_t size = max_movement_size; size > elem_size; size /= 2) {
    if (last_dst_dim_size % size == 0 && last_dst_pos % size == 0 && last_src_dim_size % size == 0
        && last_src_pos % size == 0 && last_extent % size == 0 && src_ptr % size == 0
        && dst_ptr % size == 0) {
      return size;
    }
  }
  return elem_size;
}

void SimplifyCopyNdDims(size_t num_dims, const int64_t* dst_dims, const int64_t* dst_pos,
                        const int64_t* src_dims, const int64_t* src_pos, const int64_t* extent,
                        size_t* simplified_num_dims, int64_t* simplified_dst_dims,
                        int64_t* simplified_dst_pos, int64_t* simplified_src_dims,
                        int64_t* simplified_src_pos, int64_t* simplified_extent) {
  CHECK_NE(num_dims, 0);
  size_t valid_num_dims = 0;
  FOR_RANGE(size_t, i, 0, num_dims) {
    if ((i != 0) && (dst_dims[i] == src_dims[i]) && (dst_dims[i] == extent[i]) && (src_pos[i] == 0)
        && (dst_pos[i] == 0)) {
      simplified_dst_dims[valid_num_dims - 1] *= extent[i];
      simplified_dst_pos[valid_num_dims - 1] *= extent[i];
      simplified_src_dims[valid_num_dims - 1] *= extent[i];
      simplified_src_pos[valid_num_dims - 1] *= extent[i];
      simplified_extent[valid_num_dims - 1] *= extent[i];
    } else {
      simplified_dst_dims[valid_num_dims] = dst_dims[i];
      simplified_dst_pos[valid_num_dims] = dst_pos[i];
      simplified_src_dims[valid_num_dims] = src_dims[i];
      simplified_src_pos[valid_num_dims] = src_pos[i];
      simplified_extent[valid_num_dims] = extent[i];
      valid_num_dims += 1;
    }
  }
  *simplified_num_dims = valid_num_dims;
}

constexpr size_t kMaxMovementSize = 16;
constexpr size_t kMaxNumDims = 8;

template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(Stream* stream, CopyNdKernelParams<num_dims, IndexType> params);

template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(Stream* stream, void* dst, const int64_t* dst_dims, const int64_t* dst_pos,
                  const void* src, const int64_t* src_dims, const int64_t* src_pos,
                  const int64_t* extent, size_t count) {
  CopyNdKernelParams<num_dims, IndexType> params;
  params.dst_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(dst_dims);
  params.src_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(src_dims);
  params.copy_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(extent);
  for (size_t i = 0; i < num_dims; ++i) {
    params.dst_pos[i] = dst_pos[i];
    params.src_pos[i] = src_pos[i];
  }
  params.src = src;
  params.dst = dst;
  params.count = static_cast<IndexType>(count);
  LaunchKernel<num_dims, movement_size, IndexType>(stream, params);
}

template<size_t num_dims, size_t movement_size>
void DispatchIndexType(Stream* stream, void* dst, const int64_t* dst_dims, const int64_t* dst_pos,
                       const void* src, const int64_t* src_dims, const int64_t* src_pos,
                       const int64_t* extent) {
  size_t count = 1;
  for (size_t i = 0; i < num_dims; ++i) { count *= extent[i]; }
  if (count < GetMaxVal<int32_t>()) {
    LaunchKernel<num_dims, movement_size, int32_t>(stream, dst, dst_dims, dst_pos, src, src_dims,
                                                   src_pos, extent, count);
  } else {
    LaunchKernel<num_dims, movement_size, int64_t>(stream, dst, dst_dims, dst_pos, src, src_dims,
                                                   src_pos, extent, count);
  }
}

template<size_t num_dims>
void DispatchMovementSize(Stream* stream, size_t movement_size, void* dst, const int64_t* dst_dims,
                          const int64_t* dst_pos, const void* src, const int64_t* src_dims,
                          const int64_t* src_pos, const int64_t* extent) {
  void (*func)(Stream* /*stream*/, void* /*dst*/, const int64_t* /*dst_dims*/,
               const int64_t* /*dst_pos*/, const void* /*src*/, const int64_t* /*src_dims*/,
               const int64_t* /*src_pos*/, const int64_t* /*extent*/) = nullptr;
  if (movement_size == 1) {
    func = DispatchIndexType<num_dims, 1>;
  } else if (movement_size == 2) {
    func = DispatchIndexType<num_dims, 2>;
  } else if (movement_size == 4) {
    func = DispatchIndexType<num_dims, 4>;
  } else if (movement_size == 8) {
    func = DispatchIndexType<num_dims, 8>;
  } else if (movement_size == 16) {
    func = DispatchIndexType<num_dims, 16>;
  } else {
    UNIMPLEMENTED();
  }
  func(stream, dst, dst_dims, dst_pos, src, src_dims, src_pos, extent);
}

void LaunchWithSimplified(Stream* stream, size_t movement_size, size_t num_dims, void* dst,
                          const int64_t* dst_dims, const int64_t* dst_pos, const void* src,
                          const int64_t* src_dims, const int64_t* src_pos, const int64_t* extent) {
  void (*func)(Stream* /*stream*/, size_t /*movement_size*/, void* /*dst*/,
               const int64_t* /*dst_dims*/, const int64_t* /*dst_pos*/, const void* /*src*/,
               const int64_t* /*src_dims*/, const int64_t* /*src_pos*/, const int64_t* /*extent*/) =
      nullptr;
  if (num_dims == 1) {
    func = DispatchMovementSize<1>;
  } else if (num_dims == 2) {
    func = DispatchMovementSize<2>;
  } else if (num_dims == 3) {
    func = DispatchMovementSize<3>;
  } else if (num_dims == 4) {
    func = DispatchMovementSize<4>;
  } else if (num_dims == 5) {
    func = DispatchMovementSize<5>;
  } else if (num_dims == 6) {
    func = DispatchMovementSize<6>;
  } else if (num_dims == 7) {
    func = DispatchMovementSize<7>;
  } else if (num_dims == 8) {
    func = DispatchMovementSize<8>;
  } else {
    UNIMPLEMENTED();
  }
  func(stream, movement_size, dst, dst_dims, dst_pos, src, src_dims, src_pos, extent);
}

template<size_t max_movement_size>
void SimplifyCopyNd(size_t num_dims, const int64_t* dst_dims, const int64_t* dst_pos,
                    const int64_t* src_dims, const int64_t* src_pos, const int64_t* extent,
                    size_t* simplified_num_dims, int64_t* simplified_dst_dims,
                    int64_t* simplified_dst_pos, int64_t* simplified_src_dims,
                    int64_t* simplified_src_pos, int64_t* simplified_extent, size_t elem_size,
                    void* dst, const void* src, size_t* movement_size) {
  SimplifyCopyNdDims(num_dims, dst_dims, dst_pos, src_dims, src_pos, extent, simplified_num_dims,
                     simplified_dst_dims, simplified_dst_pos, simplified_src_dims,
                     simplified_src_pos, simplified_extent);
  *movement_size = GetMovementSize<max_movement_size>(
      elem_size, *simplified_num_dims, dst, simplified_dst_dims, simplified_dst_pos, src,
      simplified_src_dims, simplified_src_pos, simplified_extent);
  size_t movement_elem_num = *movement_size / elem_size;
  simplified_dst_dims[*simplified_num_dims - 1] /= movement_elem_num;
  simplified_dst_pos[*simplified_num_dims - 1] /= movement_elem_num;
  simplified_src_dims[*simplified_num_dims - 1] /= movement_elem_num;
  simplified_src_pos[*simplified_num_dims - 1] /= movement_elem_num;
  simplified_extent[*simplified_num_dims - 1] /= movement_elem_num;
}

void SimplifyThenLaunch(Stream* stream, DataType data_type, size_t num_dims, void* dst,
                        const int64_t* dst_dims, const int64_t* dst_pos, const void* src,
                        const int64_t* src_dims, const int64_t* src_pos, const int64_t* extent) {
  CHECK_GT(num_dims, 0) << "num_dims must greater than 0";
  CHECK_LE(num_dims, kMaxNumDims);
  size_t simplified_num_dims = 0;
  int64_t simplified_dst_dims[kMaxNumDims];
  int64_t simplified_dst_pos[kMaxNumDims];
  int64_t simplified_src_dims[kMaxNumDims];
  int64_t simplified_src_pos[kMaxNumDims];
  int64_t simplified_extent[kMaxNumDims];
  size_t movement_size;
  SimplifyCopyNd<kMaxMovementSize>(num_dims, dst_dims, dst_pos, src_dims, src_pos, extent,
                                   &simplified_num_dims, simplified_dst_dims, simplified_dst_pos,
                                   simplified_src_dims, simplified_src_pos, simplified_extent,
                                   GetSizeOfDataType(data_type), dst, src, &movement_size);
  LaunchWithSimplified(stream, movement_size, simplified_num_dims, dst, simplified_dst_dims,
                       simplified_dst_pos, src, simplified_src_dims, simplified_src_pos,
                       simplified_extent);
}

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_COMMON_COPY_ND_H_
