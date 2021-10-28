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
#ifndef ONEFLOW_CORE_PRIMITIVE_BROADCAST_ELEMENTWISE_BINARY_H_
#define ONEFLOW_CORE_PRIMITIVE_BROADCAST_ELEMENTWISE_BINARY_H_

#include "oneflow/core/primitive/include/primitive.h"
#include "oneflow/core/primitive/include/binary_op.h"
#include "oneflow/core/common/scalar.h"

namespace oneflow {

namespace primitive {

class BroadcastElementwiseBinary : public Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinary);
  BroadcastElementwiseBinary() = default;
  ~BroadcastElementwiseBinary() override = default;

  virtual void Launch(StreamContext* stream_ctx, size_t num_src0_dims, const int64_t* src0_dims,
                      const void* src0, size_t num_src1_dims, const int64_t* src1_dims,
                      const void* src1, void* dst) = 0;
  virtual void Launch(StreamContext* stream_ctx, Scalar src0, size_t num_src1_dims,
                      const int64_t* src1_dims, const void* src1, void* dst) = 0;
  virtual void Launch(StreamContext* stream_ctx, size_t num_src0_dims, const int64_t* src0_dims,
                      const void* src0, Scalar src1, void* dst) = 0;
};

class BroadcastElementwiseBinaryFactory : public Factory<BroadcastElementwiseBinary> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinaryFactory);
  BroadcastElementwiseBinaryFactory() = default;
  ~BroadcastElementwiseBinaryFactory() override = default;

  virtual std::unique_ptr<BroadcastElementwiseBinary> New(BinaryOp op, DataType src_type,
                                                          DeviceType dst_type,
                                                          size_t max_num_dims) = 0;
};

}  // namespace primitive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_BROADCAST_ELEMENTWISE_BINARY_H_
