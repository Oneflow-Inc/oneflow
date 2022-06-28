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

#define BINARY_ACTIVATION_BACKWARD_OP_SEQ                     \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kEluBackwardWithDyX)         \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kCeluBackwardWithDyX)        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGeluBackwardWithDyX)        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kHardswishBackwardWithDyX)   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kHardsigmoidBackwardWithDyX) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kHardshrinkBackwardWithDyY)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kHardtanhBackwardWithDyY)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLeakyReluBackwardWithDyX)   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMishBackwardWithDyX)        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kReluBackwardWithDyY)        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSeluBackwardWithDyX)        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSiluBackwardWithDyX)        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSoftsignBackwardWithDyX)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSoftplusBackwardWithDyX)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSoftshrinkBackwardWithDyY)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kTanhBackwardWithDyX)        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kThresholdBackwardWithDyX)

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_COMMON_BROADCAST_ELEMENTWISE_BINARY
