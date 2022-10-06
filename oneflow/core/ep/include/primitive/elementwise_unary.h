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
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_ELEMENTWISE_UNARY_H_
#define ONEFLOW_CORE_EP_PRIMITIVE_ELEMENTWISE_UNARY_H_

#include "oneflow/core/common/scalar.h"
#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/unary_op.h"

namespace oneflow {

namespace ep {
namespace primitive {

class ElementwiseUnary : public Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnary);
  ElementwiseUnary() = default;
  ~ElementwiseUnary() override = default;

  virtual void Launch(Stream* stream, const void* src, void* dst, size_t count) = 0;
};

class ElementwiseUnaryFactory : public Factory<ElementwiseUnary> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryFactory);
  ElementwiseUnaryFactory() = default;
  ~ElementwiseUnaryFactory() override = default;

  virtual std::unique_ptr<ElementwiseUnary> New(UnaryOp op, DataType src_type,
                                                DataType dst_type) = 0;

  virtual std::unique_ptr<ElementwiseUnary> New(UnaryOp op, DataType src_type, DataType dst_type,
                                                Scalar attr0) = 0;

  virtual std::unique_ptr<ElementwiseUnary> New(UnaryOp op, DataType src_type, DataType dst_type,
                                                Scalar attr0, Scalar attr1) = 0;
};

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_ELEMENTWISE_UNARY_H_
