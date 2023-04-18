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
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"
#include "oneflow/cambricon/ep/primitive/elementwise_unary.h"
#include "oneflow/cambricon/ep/mlu_stream.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace mlu {

template<UnaryOp unary_op>
std::unique_ptr<ElementwiseUnary> NewElementwiseUnary(Scalar attr0, Scalar attr1, DataType src_type,
                                                      DataType dst_type) {
  return std::unique_ptr<ElementwiseUnary>(
      new ElementwiseUnaryImpl<unary_op>(attr0, attr1, src_type, dst_type));
}

class ElementwiseUnaryFactoryImpl : public ElementwiseUnaryFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryFactoryImpl);
  ElementwiseUnaryFactoryImpl() = default;
  ~ElementwiseUnaryFactoryImpl() override = default;

  std::unique_ptr<ElementwiseUnary> New(UnaryOp unary_op, DataType src_type,
                                        DataType dst_dtype) override {
    return New(unary_op, src_type, dst_dtype, Scalar(), Scalar());
  }

  std::unique_ptr<ElementwiseUnary> New(UnaryOp unary_op, DataType src_type, DataType dst_dtype,
                                        Scalar attr0) override {
    return New(unary_op, src_type, dst_dtype, attr0, Scalar());
  }

  std::unique_ptr<ElementwiseUnary> New(UnaryOp unary_op, DataType src_type, DataType dst_dtype,
                                        Scalar attr0, Scalar attr1) override {
#define MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY(unary_op, dtype_pair)                   \
  {std::make_tuple(unary_op, OF_PP_PAIR_SECOND(dtype_pair), OF_PP_PAIR_SECOND(dtype_pair)), \
   NewElementwiseUnary<unary_op>},

#define MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY(unary_op, src_type_pair, dst_dtype_pair)  \
  {std::make_tuple(unary_op, OF_PP_PAIR_SECOND(src_type_pair), OF_PP_PAIR_SECOND(dst_dtype_pair)), \
   NewElementwiseUnary<unary_op>},

    static const std::map<
        std::tuple<UnaryOp, DataType, DataType>,
        std::function<std::unique_ptr<ElementwiseUnary>(Scalar, Scalar, DataType, DataType)>>
        new_elementwise_unary_handle{
            // For Float Type OP
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY, MLU_UNARY_FLOATING_MATH_OP_SEQ,
                CPU_PRIMITIVE_FLOATING_TYPE_SEQ CPU_PRIMITIVE_FLOAT16_TYPE_SEQ)
            // For Utils OP
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                             MLU_UNARY_UTILS_OP_SEQ, UTIL_OPS_DATA_TYPE_SEQ,
                                             CPU_PRIMITIVE_BOOL_TYPE_SEQ)};

#undef MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY
#undef MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY

    const auto it =
        new_elementwise_unary_handle.find(std::make_tuple(unary_op, src_type, dst_dtype));
    if (it != new_elementwise_unary_handle.end()) {
      return it->second(attr0, attr1, src_type, dst_dtype);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kMLU, ElementwiseUnaryFactory, ElementwiseUnaryFactoryImpl);

}  // namespace mlu
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
