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
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_BROADCAST_ELEMENTWISE_UNARY_H_
#define ONEFLOW_CORE_EP_PRIMITIVE_BROADCAST_ELEMENTWISE_UNARY_H_

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/unary_op.h"
#include "oneflow/core/common/scalar.h"

namespace oneflow {

namespace ep {
namespace primitive {

class BroadcastElementwiseUnary : public Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseUnary);
  BroadcastElementwiseUnary() = default;
  ~BroadcastElementwiseUnary() override = default;

  virtual void Launch(Stream* stream, size_t num_src_dims, const int64_t* src_dims,
                      const int64_t* src_strides, const void* src, size_t num_dst_dims,
                      const int64_t* dst_dims, const int64_t* dst_strides, void* dst) = 0;

  virtual void Launch(Stream* stream, size_t num_src_dims, const int64_t* src_dims, const void* src,
                      size_t num_dst_dims, const int64_t* dst_dims, void* dst) = 0;
};

class BroadcastElementwiseUnaryFactory : public Factory<BroadcastElementwiseUnary> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseUnaryFactory);
  BroadcastElementwiseUnaryFactory() = default;
  ~BroadcastElementwiseUnaryFactory() override = default;

  virtual std::unique_ptr<BroadcastElementwiseUnary> New(UnaryOp op, DataType src_type,
                                                         DataType dst_type,
                                                         size_t max_num_dims) = 0;

  virtual std::unique_ptr<BroadcastElementwiseUnary> New(UnaryOp op, DataType src_type,
                                                         DataType dst_type, size_t max_num_dims,
                                                         Scalar attr0) = 0;

  virtual std::unique_ptr<BroadcastElementwiseUnary> New(UnaryOp op, DataType src_type,
                                                         DataType dst_type, size_t max_num_dims,
                                                         Scalar attr0, Scalar attr1) = 0;
};

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_BROADCAST_ELEMENTWISE_UNARY_H_
