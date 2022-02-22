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
#ifndef ONEFLOW_CORE_EP_COMMON_PRIMITIVE_PERMUTE_IMPL_H_
#define ONEFLOW_CORE_EP_COMMON_PRIMITIVE_PERMUTE_IMPL_H_

#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/ep/common/primitive/permute.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace permute {

namespace internal {

namespace {

template<size_t num_dims, typename IndexType>
struct PermuteKernelParams {
  NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> dst_index_helper;
  int permutation[num_dims]{};
  IndexType count{};
  const void* src{};
  void* dst{};
};

constexpr size_t kMaxMovementSize = 16;
constexpr size_t kMaxNumDims = 8;

template<size_t num_dims, typename IndexType>
PermuteKernelParams<num_dims, IndexType> MakePermuteParams(const int64_t* src_dims, const void* src,
                                                           const int* permutation, void* dst,
                                                           size_t count) {
  PermuteKernelParams<num_dims, IndexType> params;
  params.src_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(src_dims);
  int64_t dst_dims[num_dims];
  for (size_t i = 0; i < num_dims; ++i) { dst_dims[i] = src_dims[permutation[i]]; }
  params.dst_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(dst_dims);
  for (size_t i = 0; i < num_dims; ++i) { params.permutation[i] = permutation[i]; }
  params.src = src;
  params.dst = dst;
  params.count = static_cast<IndexType>(count);
  return params;
}

template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(Stream* stream, const int64_t* src_dims, const void* src, const int* permutation,
                  void* dst, size_t count);

template<size_t num_dims, size_t movement_size>
void DispatchIndexType(Stream* stream, const int64_t* src_dims, const void* src,
                       const int* permutation, void* dst) {
  size_t count = 1;
  for (size_t i = 0; i < num_dims; ++i) { count *= src_dims[i]; }
  if (count < GetMaxVal<int32_t>()) {
    LaunchKernel<num_dims, movement_size, int32_t>(stream, src_dims, src, permutation, dst, count);
  } else {
    LaunchKernel<num_dims, movement_size, int64_t>(stream, src_dims, src, permutation, dst, count);
  }
}

template<size_t num_dims>
void DispatchMovementSize(Stream* stream, size_t movement_size, const int64_t* src_dims,
                          const void* src, const int* permutation, void* dst) {
  void (*func)(Stream* /*stream*/, const int64_t* /*src_dims*/, const void* /*src*/,
               const int* /*permutation*/, void* /*dst*/) = nullptr;
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
  func(stream, src_dims, src, permutation, dst);
}

void LaunchWithSimplified(Stream* stream, size_t movement_size, size_t num_dims,
                          const int64_t* src_dims, const void* src, const int* permutation,
                          void* dst) {
  void (*func)(Stream* /*stream*/, size_t /*movement_size*/, const int64_t* /*src_dims*/,
               const void* /*src*/, const int* /*permutation*/, void* /*dst*/) = nullptr;
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
  func(stream, movement_size, src_dims, src, permutation, dst);
}

void SimplifyThenLaunch(Stream* stream, DataType data_type, size_t num_dims,
                        const int64_t* src_dims, const void* src, const int* permutation,
                        void* dst) {
  CHECK_LE(num_dims, kMaxNumDims);
  CHECK_GT(num_dims, 0);
  size_t simplified_num_dims = 0;
  int64_t simplified_src_dims[kMaxNumDims];
  int simplified_permutation[kMaxNumDims];
  size_t movement_size = 0;
  SimplifyPermutation<kMaxNumDims, kMaxMovementSize>(
      num_dims, src_dims, permutation, &simplified_num_dims, simplified_src_dims,
      simplified_permutation, GetSizeOfDataType(data_type), src, dst, &movement_size);
  LaunchWithSimplified(stream, movement_size, simplified_num_dims, simplified_src_dims, src,
                       simplified_permutation, dst);
}

}  // namespace

}  // namespace internal

}  // namespace permute

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_COMMON_PRIMITIVE_PERMUTE_IMPL_H_
