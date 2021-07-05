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

#include "oneflow/core/functional/impl/unary_functor.h"

#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/user/ops/math_unary_elementwise_seq.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

#define UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name)                          \
  class class_name##Functor : public UnaryFunctor {                                  \
   public:                                                                           \
    class_name##Functor() {                                                          \
      op_ = CHECK_JUST(one::OpBuilder(op_type_name).Input("x").Output("y").Build()); \
    }                                                                                \
  };

OF_PP_FOR_EACH_TUPLE(UNARY_ELEMENTWISE_FUNCTOR, MATH_UNARY_ELEMENTWISE_FUNC_SEQ);

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::AbsFunctor>("Abs");
  m.add_functor<impl::AcosFunctor>("Acos");
  m.add_functor<impl::AcoshFunctor>("Acosh");
  m.add_functor<impl::AsinFunctor>("Asin");
  m.add_functor<impl::AsinhFunctor>("Asinh");
  m.add_functor<impl::AtanFunctor>("Atan");
  m.add_functor<impl::AtanhFunctor>("Atanh");
  m.add_functor<impl::CeilFunctor>("Ceil");
  m.add_functor<impl::CosFunctor>("Cos");
  m.add_functor<impl::CoshFunctor>("Cosh");
  m.add_functor<impl::ErfFunctor>("Erf");
  m.add_functor<impl::ErfcFunctor>("Erfc");
  m.add_functor<impl::ExpFunctor>("Exp");
  m.add_functor<impl::Expm1Functor>("Expm1");
  m.add_functor<impl::FloorFunctor>("Floor");
  m.add_functor<impl::LgammaFunctor>("Lgamma");
  m.add_functor<impl::LogFunctor>("Log");
  m.add_functor<impl::Log1pFunctor>("Log1p");
  m.add_functor<impl::LogSigmoidFunctor>("LogSigmoid");
  m.add_functor<impl::NegativeFunctor>("Negative");
  m.add_functor<impl::ReciprocalFunctor>("Reciprocal");
  m.add_functor<impl::ReciprocalNoNanFunctor>("ReciprocalNoNan");
  m.add_functor<impl::RintFunctor>("Rint");
  m.add_functor<impl::RoundFunctor>("Round");
  m.add_functor<impl::RsqrtFunctor>("Rsqrt");
  m.add_functor<impl::SigmoidFunctor>("Sigmoid");
  m.add_functor<impl::SignFunctor>("Sign");
  m.add_functor<impl::SinFunctor>("Sin");
  m.add_functor<impl::SinhFunctor>("Sinh");
  m.add_functor<impl::SoftplusFunctor>("Softplus");
  m.add_functor<impl::SqrtFunctor>("Sqrt");
  m.add_functor<impl::SquareFunctor>("Square");
  m.add_functor<impl::TanFunctor>("Tan");
  m.add_functor<impl::TanhFunctor>("Tanh");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
