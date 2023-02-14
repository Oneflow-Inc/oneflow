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
#include "oneflow/core/ep/cuda/primitive/broadcast_elementwise_binary.cuh"

namespace oneflow {

namespace ep {
namespace primitive {
namespace broadcast_elementwise_binary {

#define INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_BITWISE_ENTRY(binary_op, data_type_pair) \
  template std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary<         \
      binary_op, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(data_type_pair)>(         \
      Scalar attr0, Scalar attr1);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_BITWISE_ENTRY,
                                 BINARY_BITWISE_OP_SEQ,
                                 CUDA_PRIMITIVE_INT_TYPE_SEQ CUDA_PRIMITIVE_BOOL_TYPE_SEQ);

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
