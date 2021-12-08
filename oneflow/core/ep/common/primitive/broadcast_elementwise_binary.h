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
#ifndef ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_ELEMENTWISE_BINARY
#define ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_ELEMENTWISE_BINARY

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/binary_op.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/ep/common/primitive/util.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace broadcast_elementwise_binary {

constexpr size_t kMaxNumDims = 8;

inline void CheckInplace(size_t num_dims, const int64_t* src0_dims, const void* src0,
                         const int64_t* src1_dims, const void* src1, const int64_t* dst_dims,
                         const void* dst) {
  for (int64_t i = 0; i < num_dims; ++i) {
    if (src0 == dst) { CHECK_EQ(src0_dims[i], dst_dims[i]); }
    if (src1 == dst) { CHECK_EQ(src1_dims[i], dst_dims[i]); }
  }
}

inline bool IsDimsEquals(size_t num_src0_dims, const int64_t* src0_dims, size_t num_src1_dims,
                         const int64_t* src1_dims) {
  if (num_src0_dims != num_src1_dims) { return false; }
  for (size_t i = 0; i < num_src1_dims; ++i) {
    if (src0_dims[i] != src1_dims[i]) { return false; }
  }
  return true;
}

inline void SimplifyDims(size_t num_src0_dims, const int64_t* src0_dims, size_t num_src1_dims,
                         const int64_t* src1_dims, size_t* simplified_num_dims,
                         int64_t* simplified_src0_dims, int64_t* simplified_src1_dims,
                         int64_t* simplified_dst_dims) {
  size_t src0_count = GetElementCount(num_src0_dims, src0_dims);
  size_t src1_count = GetElementCount(num_src1_dims, src1_dims);
  if (src0_count == 1 || src1_count == 1) {
    *simplified_num_dims = 1;
    simplified_src0_dims[0] = src0_count;
    simplified_src1_dims[0] = src1_count;
    simplified_dst_dims[0] = std::max(src0_count, src1_count);
    return;
  }
  const size_t num_max_dims = std::max(num_src0_dims, num_src1_dims);
  auto MakeGetDim = [num_max_dims](size_t num_dims, const int64_t* dims) {
    const int64_t num_padding_dims = num_max_dims - num_dims;
    return [num_padding_dims, dims](size_t index) {
      return index < num_padding_dims ? 1 : dims[index - num_padding_dims];
    };
  };
  *simplified_num_dims = 0;
  bool prev_broadcast_src0 = false;
  bool prev_broadcast_src1 = false;
  auto GetSrc0Dim = MakeGetDim(num_src0_dims, src0_dims);
  auto GetSrc1Dim = MakeGetDim(num_src1_dims, src1_dims);
  for (int64_t i = 0; i < num_max_dims; ++i) {
    const int64_t src0_dim = GetSrc0Dim(i);
    const int64_t src1_dim = GetSrc1Dim(i);
    const int64_t broadcast_dim = std::max(src0_dim, src1_dim);
    CHECK_GT(broadcast_dim, 0);
    const bool broadcast_src0 = (src0_dim == 1);
    const bool broadcast_src1 = (src1_dim == 1);
    CHECK((src0_dim == broadcast_dim) || broadcast_src0);
    CHECK((src1_dim == broadcast_dim) || broadcast_src1);
    if (broadcast_dim == 1) {
      continue;
    } else if (*simplified_num_dims != 0
               && (prev_broadcast_src0 == broadcast_src0
                   && prev_broadcast_src1 == broadcast_src1)) {
      simplified_src0_dims[*simplified_num_dims - 1] *= src0_dim;
      simplified_src1_dims[*simplified_num_dims - 1] *= src1_dim;
      simplified_dst_dims[*simplified_num_dims - 1] *= broadcast_dim;
    } else {
      simplified_src0_dims[*simplified_num_dims] = src0_dim;
      simplified_src1_dims[*simplified_num_dims] = src1_dim;
      simplified_dst_dims[*simplified_num_dims] = broadcast_dim;
      *simplified_num_dims += 1;
      prev_broadcast_src0 = broadcast_src0;
      prev_broadcast_src1 = broadcast_src1;
    }
  }
}

#define BINARY_MATH_OP_SEQ             \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kAdd) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSub) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMul) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kDiv) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMax) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMin) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kPow)

#define BINARY_COMPARISION_OP_SEQ              \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kEqual)       \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kNotEqual)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLessThan)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLessEqual)   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGreaterThan) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGreaterEqual)

#define BINARY_LOGICAL_OP_SEQ                 \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalAnd) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalOr)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalXor)

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_ELEMENTWISE_BINARY
