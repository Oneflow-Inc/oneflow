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
#ifndef ONEFLOW_OPENCL_EP_PRIMITIVE_BROADCAST_ELEMENTWISE_BINARY_H_
#define ONEFLOW_OPENCL_EP_PRIMITIVE_BROADCAST_ELEMENTWISE_BINARY_H_

#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"

namespace oneflow {
namespace ep {
namespace primitive {
namespace opencl {

#define CL_BINARY_MATH_OP_SEQ          \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kAdd) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSub) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMul) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kDiv)

inline int64_t ComputeElementCount(size_t ndim, const int64_t* dims) {
  int64_t count = 1;
  for (int i = 0; i < ndim; ++i) { count *= dims[i]; }
  return count;
}

inline std::vector<int64_t> ComputeBroadcastShape(size_t small_num_dims, const int64_t* small_dims,
                                                  size_t large_num_dims,
                                                  const int64_t* large_dims) {
  std::vector<int64_t> dst_dims(large_num_dims);
  size_t offset = large_num_dims - small_num_dims;
  for (int i = 0; i < offset; ++i) { dst_dims[i] = large_dims[i]; }
  for (int i = offset; i < large_num_dims; ++i) {
    int64_t dim0 = large_dims[i];
    int64_t dim1 = small_dims[i - offset];
    dst_dims[i] = (dim0 > dim1) ? dim0 : dim1;
  }
  return dst_dims;
}

template<BinaryOp op>
inline std::string getBinaryOpAsString();

template<>
inline std::string getBinaryOpAsString<BinaryOp::kAdd>() {
  return "+";
}
template<>
inline std::string getBinaryOpAsString<BinaryOp::kSub>() {
  return "-";
}
template<>
inline std::string getBinaryOpAsString<BinaryOp::kMul>() {
  return "*";
}
template<>
inline std::string getBinaryOpAsString<BinaryOp::kDiv>() {
  return "/";
}

template<typename T>
inline std::string getCppDTypeAsString();

template<>
inline std::string getCppDTypeAsString<float>() {
  return "float";
}

template<BinaryOp binary_op, typename Src, typename Dst>
std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary(Scalar attr0,
                                                                          Scalar attr1);

}  // namespace opencl
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_OPENCL_EP_PRIMITIVE_BROADCAST_ELEMENTWISE_BINARY_H_
