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
#ifndef ONEFLOW_CAMBRICON_EP_PRIMITIVE_BROADCAST_ELEMENTWISE_BINARY_H_
#define ONEFLOW_CAMBRICON_EP_PRIMITIVE_BROADCAST_ELEMENTWISE_BINARY_H_

#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"

namespace oneflow {
namespace ep {
namespace primitive {
namespace mlu {

#define MLU_BINARY_LOGICAL_OP_SEQ               \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kEqual)        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kNotEqual)     \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGreaterThan)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGreaterEqual) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLessThan)     \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLessEqual)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalAnd)   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalOr)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalXor)

#define MLU_CNNL_LOGICAL_OP_SEQ                                   \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kEqual, CNNL_LOGIC_OP_EQ)        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kNotEqual, CNNL_LOGIC_OP_NE)     \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGreaterThan, CNNL_LOGIC_OP_GT)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGreaterEqual, CNNL_LOGIC_OP_GE) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLessThan, CNNL_LOGIC_OP_LT)     \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLessEqual, CNNL_LOGIC_OP_LE)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalAnd, CNNL_LOGIC_OP_AND)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalOr, CNNL_LOGIC_OP_OR)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalXor, CNNL_LOGIC_OP_XOR)

template<BinaryOp binary_op, typename Src, typename Dst>
std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary(Scalar attr0,
                                                                          Scalar attr1);

}  // namespace mlu
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_EP_PRIMITIVE_BROADCAST_ELEMENTWISE_BINARY_H_
