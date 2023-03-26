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
#ifndef ONEFLOW_CAMBRICON_EP_PRIMITIVE_UNARY_ELEMENTWISE_BINARY_H_
#define ONEFLOW_CAMBRICON_EP_PRIMITIVE_UNARY_ELEMENTWISE_BINARY_H_

#include "oneflow/core/ep/include/primitive/unary_op.h"
#include "oneflow/core/ep/include/primitive/elementwise_unary.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"
#include "oneflow/cambricon/ep/primitive/type_seq.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/cambricon/ep/mlu_device.h"

namespace oneflow {
namespace ep {
namespace primitive {
namespace mlu {

#define MLU_UNARY_FLOATING_MATH_OP_SEQ            \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kReciprocal)      \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kReciprocalNoNan) \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRsqrt)

template<UnaryOp unary_op>
class ElementwiseUnaryImpl : public ElementwiseUnary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryImpl);
  ElementwiseUnaryImpl(Scalar attr0, Scalar attr1, DataType src_dtype, DataType dst_dtype)
      : attr0(attr0), attr1(attr1), src_dtype(src_dtype), dst_dtype(dst_dtype) {}
  ~ElementwiseUnaryImpl() override = default;

  void Launch(Stream* stream, const void* src_ptr, void* dst_ptr, size_t count) override {
    CnnlTensorDescriptor input_desc, output_desc;
    std::vector<int64_t> dims = {static_cast<int64_t>(count)};
    input_desc.set(1, dims.data(), ConvertToCnnlDataType(src_dtype));
    output_desc.set(1, dims.data(), ConvertToCnnlDataType(dst_dtype));

    auto* cnnl_handle = stream->As<ep::MluStream>()->cnnl_handle();

    if constexpr (unary_op == UnaryOp::kReciprocal) {
      OF_CNNL_CHECK(
          cnnlReciprocal(cnnl_handle, input_desc.desc(), src_ptr, output_desc.desc(), dst_ptr));
    } else if constexpr (unary_op == UnaryOp::kReciprocalNoNan) {
      OF_CNNL_CHECK(
          cnnlReciprocal(cnnl_handle, input_desc.desc(), src_ptr, output_desc.desc(), dst_ptr));
      OF_CNNL_CHECK(cnnlNanToNum(cnnl_handle, output_desc.desc(), static_cast<const void*>(dst_ptr),
                                 0, 0, 0, output_desc.desc(), dst_ptr));
    } else if constexpr (unary_op == UnaryOp::kRsqrt) {
      OF_CNNL_CHECK(cnnlRsqrt_v2(cnnl_handle, CNNL_COMPUTATION_HIGH_PRECISION, input_desc.desc(),
                                 src_ptr, output_desc.desc(), dst_ptr));
    } else {
      UNIMPLEMENTED();
    }
  }

 protected:
  Scalar attr0, attr1;
  DataType src_dtype, dst_dtype;
};

}  // namespace mlu
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_EP_PRIMITIVE_UNARY_ELEMENTWISE_BINARY_H_
