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
#include "oneflow/core/ep/common/primitive/broadcast_elementwise_binary.h"
#include "oneflow/opencl/ep/primitive/broadcast_elementwise_binary.h"
#include "oneflow/opencl/ep/primitive/type_seq.h"

namespace oneflow {
namespace ep {
namespace primitive {
namespace opencl {

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

  std::unique_ptr<BroadcastElementwiseBinary> New(BinaryOp op, DataType src_type, DataType dst_type,
                                                  size_t max_num_dims, Scalar attr0,
                                                  Scalar attr1) override {
    if (max_num_dims > kMaxNumDims) { return nullptr; }

#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY(binary_op, data_type_pair) \
  {std::make_tuple(binary_op, OF_PP_PAIR_SECOND(data_type_pair),                    \
                   OF_PP_PAIR_SECOND(data_type_pair)),                              \
   NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(data_type_pair),       \
                                 OF_PP_PAIR_FIRST(data_type_pair)>},

    static const std::map<
        std::tuple<BinaryOp, DataType, DataType>,
        std::function<std::unique_ptr<BroadcastElementwiseBinary>(Scalar, Scalar)>>
        new_broadcast_elementwise_binary_handle{
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY,
                                             CL_BINARY_MATH_OP_SEQ, CL_PRIMITIVE_ALL_TYPE_SEQ)};

#undef MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY

    const auto it =
        new_broadcast_elementwise_binary_handle.find(std::make_tuple(op, src_type, dst_type));
    if (it != new_broadcast_elementwise_binary_handle.end()) {
      return it->second(attr0, attr1);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kOpenCL, BroadcastElementwiseBinaryFactory,
                           BroadcastElementwiseBinaryFactoryImpl);

}  // namespace

}  // namespace opencl
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
