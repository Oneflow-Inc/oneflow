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
#include "oneflow/core/primitive/common/unary_op_utils.h"
#include "oneflow/core/primitive/cpu/type_seq.h"

namespace oneflow {

namespace primitive {

namespace {

template<UnaryOpList unary_enum, typename T>
class ElementwiseUnaryOpImpl : public ElementwiseUnaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryOpImpl);
  ElementwiseUnaryOpImpl() = default;
  ~ElementwiseUnaryOpImpl() override = default;

  void Launch(StreamContext* stream_ctx, size_t count, void* dst_ptr,
              const void* src_ptr) override {
    T* dst = reinterpret_cast<T*>(dst_ptr);
    const T* src = reinterpret_cast<const T*>(src_ptr);
    for (size_t i = 0; i < count; ++i) {
      dst[i] = UnaryFunctor<DeviceType::kCPU, unary_enum, T>()(src[i]);
    }
  }
};

template<UnaryOpList unary_enum, typename T>
std::unique_ptr<ElementwiseUnaryOp> NewElementwiseUnaryOp() {
  return std::unique_ptr<ElementwiseUnaryOp>(new ElementwiseUnaryOpImpl<unary_enum, T>());
}

class ElementwiseUnaryOpFactoryImpl : public ElementwiseUnaryOpFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryOpFactoryImpl);
  ElementwiseUnaryOpFactoryImpl() = default;
  ~ElementwiseUnaryOpFactoryImpl() override = default;

  std::unique_ptr<ElementwiseUnaryOp> New(UnaryOpList op_enum, DataType dtype) override {
#define MAKE_NEW_UNARYOP_ENTRY(op_pair, dtype_pair)                           \
  {std::make_pair(OF_PP_PAIR_SECOND(op_pair), OF_PP_PAIR_SECOND(dtype_pair)), \
   NewElementwiseUnaryOp<OF_PP_PAIR_SECOND(op_pair), OF_PP_PAIR_FIRST(dtype_pair)>},

    static const std::map<std::pair<UnaryOpList, DataType>,
                          std::function<std::unique_ptr<ElementwiseUnaryOp>()>>
        new_elementwise_unary_op_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
            MAKE_NEW_UNARYOP_ENTRY, PRIMITIVE_UNARY_OP_SEQ, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)};

#undef MAKE_NEW_UNARYOP_ENTRY

    const auto it = new_elementwise_unary_op_handle.find(std::make_pair(op_enum, dtype));
    if (it != new_elementwise_unary_op_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, ElementwiseUnaryOpFactory,
                           ElementwiseUnaryOpFactoryImpl);

}  // namespace
}  // namespace primitive
}  // namespace oneflow