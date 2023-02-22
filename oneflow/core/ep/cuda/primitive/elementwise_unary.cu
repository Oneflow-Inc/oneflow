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
#include "oneflow/core/ep/common/primitive/elementwise_unary.h"
#include "oneflow/core/ep/cuda/primitive/unary_functor.cuh"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<UnaryOp unary_op, typename Src, typename Dst>
class ElementwiseUnaryImpl : public ElementwiseUnary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryImpl);
  ElementwiseUnaryImpl(Scalar attr0, Scalar attr1) : attr0(attr0), attr1(attr1) {}
  ~ElementwiseUnaryImpl() override = default;

  void Launch(Stream* stream, const void* src, void* dst, size_t count) override {
    auto* cuda_stream = stream->As<CudaStream>();
    auto functor = UnaryFunctor<DeviceType::kCUDA, unary_op, Dst, Src>(attr0, attr1);
    OF_CUDA_CHECK((cuda::elementwise::Unary<decltype(functor), Dst, Src>(
        functor, count, reinterpret_cast<Dst*>(dst), reinterpret_cast<const Src*>(src),
        cuda_stream->cuda_stream())));
  }

 protected:
  Scalar attr0, attr1;
};

template<UnaryOp unary_op, typename Src, typename Dst>
std::unique_ptr<ElementwiseUnary> NewElementwiseUnary(Scalar attr0, Scalar attr1) {
  return std::unique_ptr<ElementwiseUnary>(
      new ElementwiseUnaryImpl<unary_op, Src, Dst>(attr0, attr1));
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
   NewElementwiseUnary<unary_op, OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(dtype_pair)>},

#define MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY(unary_op, src_type_pair, dst_dtype_pair)  \
  {std::make_tuple(unary_op, OF_PP_PAIR_SECOND(src_type_pair), OF_PP_PAIR_SECOND(dst_dtype_pair)), \
   NewElementwiseUnary<unary_op, OF_PP_PAIR_FIRST(src_type_pair),                                  \
                       OF_PP_PAIR_FIRST(dst_dtype_pair)>},

    static const std::map<std::tuple<UnaryOp, DataType, DataType>,
                          std::function<std::unique_ptr<ElementwiseUnary>(Scalar, Scalar)>>
        new_elementwise_unary_handle{
            // For All Type OP
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                             UNARY_MATH_OP_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ)
            // For Float Type OP
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                             UNARY_FLOATING_MATH_OP_SEQ,
                                             CUDA_PRIMITIVE_FLOATING_TYPE_SEQ)

            // For Int Type OP
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                             UNARY_INT_MATH_OP_SEQ, CUDA_PRIMITIVE_INT_TYPE_SEQ)

            // For Utils OP
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                             UNARY_UTILS_OP_SEQ, UTIL_OPS_DATA_TYPE_SEQ,
                                             CUDA_PRIMITIVE_BOOL_TYPE_SEQ)

            // For Logical OP
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                             UNARY_LOGICAL_OP_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ,
                                             CUDA_PRIMITIVE_BOOL_TYPE_SEQ)

            // For bitwise op
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY, UNARY_BITWISE_OP_SEQ,
                CUDA_PRIMITIVE_INT_TYPE_SEQ CUDA_PRIMITIVE_BOOL_TYPE_SEQ)};

#undef MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY

#undef MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY
    const auto it =
        new_elementwise_unary_handle.find(std::make_tuple(unary_op, src_type, dst_dtype));
    if (it != new_elementwise_unary_handle.end()) {
      return it->second(attr0, attr1);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, ElementwiseUnaryFactory, ElementwiseUnaryFactoryImpl);

}  // namespace
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
