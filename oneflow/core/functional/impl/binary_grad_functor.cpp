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

#include "oneflow/core/functional/impl/binary_functor.h"

#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/user/ops/math_binary_elementwise_seq.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

#define BINARY_ELEMENTWISE_GRAD_FUNCTOR(op_type_name, class_name, base)           \
  class class_name##XGradFunctor : public base {                                  \
   public:                                                                        \
    class_name##XGradFunctor() {                                                  \
      op_ = CHECK_JUST(one::OpBuilder(std::string("") + op_type_name + "_x_grad") \
                           .Input("x")                                            \
                           .Input("y")                                            \
                           .Input("dz")                                           \
                           .Output("dx")                                          \
                           .Build());                                             \
    }                                                                             \
  };                                                                              \
  class class_name##YGradFunctor : public base {                                  \
   public:                                                                        \
    class_name##YGradFunctor() {                                                  \
      op_ = CHECK_JUST(one::OpBuilder(std::string("") + op_type_name + "_y_grad") \
                           .Input("x")                                            \
                           .Input("y")                                            \
                           .Input("dz")                                           \
                           .Output("dy")                                          \
                           .Build());                                             \
    }                                                                             \
  };

#define INSTANTIAT_BINARY_ELEMENTWISE_GRAD_FUNCTOR(op_type_name, class_name) \
  BINARY_ELEMENTWISE_GRAD_FUNCTOR(op_type_name, class_name, BinaryGradFunctor);

OF_PP_FOR_EACH_TUPLE(INSTANTIAT_BINARY_ELEMENTWISE_GRAD_FUNCTOR, MATH_BINARY_ELEMENTWISE_FUNC_SEQ);
}  // namespace impl

using namespace impl;

#define ADD_BINARY_GRAD_FUNCTOR(class_name, functor_name)                            \
  m.add_functor<class_name##XGradFunctor>(std::string("") + functor_name + "XGrad"); \
  m.add_functor<class_name##YGradFunctor>(std::string("") + functor_name + "YGrad");

ONEFLOW_FUNCTION_LIBRARY(m) {
  ADD_BINARY_GRAD_FUNCTOR(Pow, "Pow");
  ADD_BINARY_GRAD_FUNCTOR(Atan2, "Atan2");
  ADD_BINARY_GRAD_FUNCTOR(FloorDiv, "FloorDiv");
  ADD_BINARY_GRAD_FUNCTOR(TruncDiv, "TruncDiv");
  ADD_BINARY_GRAD_FUNCTOR(Xdivy, "Xdivy");
  ADD_BINARY_GRAD_FUNCTOR(Xlogy, "Xlogy");
};

#undef ADD_BINARY_GRAD_FUNCTOR

}  // namespace functional
}  // namespace one
}  // namespace oneflow
