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

#include "oneflow/core/ep/include/primitive/broadcast_elementwise_unary.h"
#include "oneflow/core/ep/common/primitive/broadcast_elementwise_unary.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

namespace ep {
namespace primitive {
namespace broadcast_elementwise_unary {

namespace {

template<UnaryOp unary_op, typename Src, typename Dst>
class BroadcastElementwiseUnaryImpl : public BroadcastElementwiseUnary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseUnaryImpl);
  BroadcastElementwiseUnaryImpl(Scalar attr0, Scalar attr1) = default;
  ~BroadcastElementwiseUnaryImpl() override = default;

  void Launch(Stream* stream, size_t num_src_dims, const int64_t* src_dims,
                      const int64_t* src_strides, const void* src, size_t num_dst_dims,
                      const int64_t* dst_dims, const int64_t* dst_strides, void* dst) override {
    // TODO(yaozihang): impl
  }

 protected:
  Scalar attr0, attr1;
};

template<UnaryOp unary_op, typename Src, typename Dst>
std::unique_ptr<BroadcastElementwiseUnary> NewBroadcastElementwiseUnary(Scalar attr0,
                                                                          Scalar attr1) {
  return std::unique_ptr<BroadcastElementwiseUnary>(
      new BroadcastElementwiseUnaryImpl<unary_op, Src, Dst>(attr0, attr1));
}

class BroadcastElementwiseUnaryFactoryImpl : public BroadcastElementwiseUnaryFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseUnaryFactoryImpl);
  BroadcastElementwiseUnaryFactoryImpl() = default;
  ~BroadcastElementwiseUnaryFactoryImpl() override = default;

  std::unique_ptr<BroadcastElementwiseUnary> New(UnaryOp unary_op, DataType src_type, DataType dst_type,
                                                 size_t max_num_dims, Scalar attr0, Scalar attr1) override {
    if (max_num_dims > kMaxNumDims) { return nullptr; }
#define MAKE_NEW_SAME_DTYPE_BROADCAST_ELEMENTWISE_UNARY_ENTRY(unary_op, dtype_pair)                   \
  {std::make_tuple(unary_op, OF_PP_PAIR_SECOND(dtype_pair), OF_PP_PAIR_SECOND(dtype_pair)), \
   NewBroadcastElementwiseUnary<unary_op, OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(dtype_pair)>},

    static const std::map<std::tuple<UnaryOp, DataType, DataType>,
        std::function<std::unique_ptr<BroadcastElementwiseUnary>(Scalar, Scalar)>>
        new_broadcast_elementwise_unary_handle {
        // TODO(yaozihang): add registry for ops which use BroadcastElementwiseUnary primitive
        };

#undef MAKE_NEW_SAME_DTYPE_BROADCAST_ELEMENTWISE_UNARY_ENTRY

    const auto iter = new_broadcast_elementwise_unary_handle.find(
        std::make_tuple(unary_op, src_type, dst_type));
    if (iter != new_broadcast_elementwise_unary_handle.end()) {
      return iter->second(attr0, attr1);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, BroadcastElementwiseUnaryFactory,
                           BroadcastElementwiseUnaryFactoryImpl);

}  // namespace
}  // namespace broadcast_elementwise_unary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
