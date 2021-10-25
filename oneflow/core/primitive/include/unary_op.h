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
#ifndef ONEFLOW_CORE_PRIMITIVE_UNARY_OP_H_
#define ONEFLOW_CORE_PRIMITIVE_UNARY_OP_H_

#include "oneflow/core/primitive/include/primitive.h"

namespace oneflow {

namespace primitive {

enum class UnaryOpList:int32_t {
  kIdentity,
};
class UnaryOp: public Primitive{
  public: 
    OF_DISALLOW_COPY_AND_MOVE(UnaryOp); 
    UnaryOp() = default; 
    ~UnaryOp() override = default; 
  private: 
    virtual void Launch(StreamContext* ctx, size_t count, void* dst, const void* src); 
}; 

class UnaryOpFactory: public Factory<UnaryOp>{
  public: 
    OF_DISALLOW_COPY_AND_MOVE(UnaryOpFactory);
    UnaryOpFactory() = default;
    ~UnaryOpFactory() override = default;
  private: 
    virtual std::unique_ptr<UnaryOp> New(UnaryOpList op_enum, DataType out_dtype, DataType in_dtype) = 0; 
}; 

template<typename Out, typename In>
struct IdentityFunctor {
  explicit IdentityFunctor(){}
  void operator()(Out* dst, const In* src, size_t count) const {
    for (size_t i = 0; i < count; ++i) { dst[i] = src[i]; }
  }
};

#define CPU_PRIMITIVE_UNARY_OP_SEQ OF_PP_MAKE_TUPLE_SEQ(IdentityFunctor, UnaryOpList::kIdentity)


}  // namespace primitive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PRIMITIVE_UNARY_OP_H_
