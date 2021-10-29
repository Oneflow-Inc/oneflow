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
#include "oneflow/core/primitive/common/elementwise_unary_utils.h"
#include "oneflow/core/primitive/cpu/type_seq.h"

namespace oneflow {

namespace primitive {

namespace {

template<UnaryOp unary_enum, typename In, typename Out>
class ElementwiseUnaryImpl : public ElementwiseUnary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryImpl);
  ElementwiseUnaryImpl() = default;
  ~ElementwiseUnaryImpl() override = default;

  void Launch(StreamContext* stream_ctx, const void* src_ptr, void* dst_ptr,
              size_t count) override {
    Out* dst = reinterpret_cast<Out*>(dst_ptr);
    const In* src = reinterpret_cast<const In*>(src_ptr);
    for (size_t i = 0; i < count; ++i) {
      dst[i] = UnaryFunctor<DeviceType::kCPU, unary_enum, Out, In>()(src[i]);
    }
  }
};

template<UnaryOp unary_enum, typename In, typename Out>
std::unique_ptr<ElementwiseUnary> NewElementwiseUnary() {
  return std::unique_ptr<ElementwiseUnary>(new ElementwiseUnaryImpl<unary_enum, In, Out>());
}

class ElementwiseUnaryFactoryImpl : public ElementwiseUnaryFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryFactoryImpl);
  ElementwiseUnaryFactoryImpl() = default;
  ~ElementwiseUnaryFactoryImpl() override = default;

  std::unique_ptr<ElementwiseUnary> New(UnaryOp op_enum, DataType in_dtype,
                                        DataType out_dtype) override {
#define MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY(op_pair, dtype_pair)         \
  {std::make_tuple(OF_PP_PAIR_SECOND(op_pair), OF_PP_PAIR_SECOND(dtype_pair),    \
                   OF_PP_PAIR_SECOND(dtype_pair)),                               \
   NewElementwiseUnary<OF_PP_PAIR_SECOND(op_pair), OF_PP_PAIR_FIRST(dtype_pair), \
                       OF_PP_PAIR_FIRST(dtype_pair)>},

#define MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY(op_pair, in_dtype_pair, out_dtype_pair) \
  {std::make_tuple(OF_PP_PAIR_SECOND(op_pair), OF_PP_PAIR_SECOND(in_dtype_pair),                 \
                   OF_PP_PAIR_SECOND(out_dtype_pair)),                                           \
   NewElementwiseUnary<OF_PP_PAIR_SECOND(op_pair), OF_PP_PAIR_FIRST(in_dtype_pair),              \
                       OF_PP_PAIR_FIRST(out_dtype_pair)>},

    static const std::map<std::tuple<UnaryOp, DataType, DataType>,
                          std::function<std::unique_ptr<ElementwiseUnary>()>>
        new_elementwise_unary_handle{
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                             PRIMITIVE_SAME_DTYPE_UNARY_OP_SEQ,
                                             CPU_PRIMITIVE_NATIVE_TYPE_SEQ)

                OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                                 PRIMITIVE_OUT_INT8_DTYPE_UNARY_OP_SEQ,
                                                 CPU_PRIMITIVE_NATIVE_TYPE_SEQ,
                                                 CPU_PRIMITIVE_INT8_TYPE_SEQ)};

#undef MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY

#undef MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY

    const auto it =
        new_elementwise_unary_handle.find(std::make_tuple(op_enum, in_dtype, out_dtype));
    if (it != new_elementwise_unary_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, ElementwiseUnaryFactory, ElementwiseUnaryFactoryImpl);

}  // namespace
}  // namespace primitive
}  // namespace oneflow
