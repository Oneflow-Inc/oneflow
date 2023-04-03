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
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/ep/common/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/primitive/binary_functor.cuh"

namespace oneflow {

namespace ep {
namespace primitive {
namespace broadcast_elementwise_binary {

template<BinaryOp binary_op, typename Src, typename Dst>
std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary(Scalar attr0,
                                                                          Scalar attr1);

namespace {

class BroadcastElementwiseBinaryFactoryImpl : public BroadcastElementwiseBinaryFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinaryFactoryImpl);
  BroadcastElementwiseBinaryFactoryImpl() = default;
  ~BroadcastElementwiseBinaryFactoryImpl() override = default;

  std::unique_ptr<BroadcastElementwiseBinary> New(BinaryOp op, DataType src_type, DataType dst_type,
                                                  size_t max_num_dims) override {
    return New(op, src_type, dst_type, max_num_dims, Scalar(), Scalar());
  }

  std::unique_ptr<BroadcastElementwiseBinary> New(BinaryOp op, DataType src_type, DataType dst_type,
                                                  size_t max_num_dims, Scalar attr0) override {
    return New(op, src_type, dst_type, max_num_dims, attr0, Scalar());
  }

  std::unique_ptr<BroadcastElementwiseBinary> New(BinaryOp binary_op, DataType src_type,
                                                  DataType dst_type, size_t max_num_dims,
                                                  Scalar attr0, Scalar attr1) override {
    if (max_num_dims > kMaxNumDims) { return nullptr; }
#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY(binary_op, data_type_pair) \
  {std::make_tuple(binary_op, OF_PP_PAIR_SECOND(data_type_pair),                    \
                   OF_PP_PAIR_SECOND(data_type_pair)),                              \
   NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(data_type_pair),       \
                                 OF_PP_PAIR_FIRST(data_type_pair)>},

#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY(      \
    binary_op, src_data_type_pair, dst_data_type_pair)                            \
  {std::make_tuple(binary_op, OF_PP_PAIR_SECOND(src_data_type_pair),              \
                   OF_PP_PAIR_SECOND(dst_data_type_pair)),                        \
   NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(src_data_type_pair), \
                                 OF_PP_PAIR_FIRST(dst_data_type_pair)>},

#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_ACTIVATION_GRAD_ENTRY(binary_op, data_type_pair) \
  {std::make_tuple(binary_op, OF_PP_PAIR_SECOND(data_type_pair),                               \
                   OF_PP_PAIR_SECOND(data_type_pair)),                                         \
   NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(data_type_pair),                  \
                                 OF_PP_PAIR_FIRST(data_type_pair)>},

    static const std::map<
        std::tuple<BinaryOp, DataType, DataType>,
        std::function<std::unique_ptr<BroadcastElementwiseBinary>(Scalar, Scalar)>>
        new_broadcast_elementwise_binary_handle{
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY,
                                             BINARY_MATH_OP_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ)

                OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                    MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY,
                    BINARY_COMPARISION_OP_SEQ BINARY_LOGICAL_OP_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ,
                    CUDA_PRIMITIVE_BOOL_TYPE_SEQ)

                    OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                        MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_ACTIVATION_GRAD_ENTRY,
                        BINARY_ACTIVATION_BACKWARD_OP_SEQ, CUDA_PRIMITIVE_FLOATING_TYPE_SEQ)

                        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                            MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_ACTIVATION_GRAD_ENTRY,
                            BINARY_MATH_BACKWARD_OP_SEQ, CUDA_PRIMITIVE_FLOATING_TYPE_SEQ)

                            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                                MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY,
                                BINARY_BITWISE_OP_SEQ,
                                CUDA_PRIMITIVE_INT_TYPE_SEQ CUDA_PRIMITIVE_BOOL_TYPE_SEQ)};

#undef MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY
#undef MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY

    const auto it = new_broadcast_elementwise_binary_handle.find(
        std::make_tuple(binary_op, src_type, dst_type));
    if (it != new_broadcast_elementwise_binary_handle.end()) {
      return it->second(attr0, attr1);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, BroadcastElementwiseBinaryFactory,
                           BroadcastElementwiseBinaryFactoryImpl);
}  // namespace
}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
