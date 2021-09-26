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
#define INPLACEABLE_UNARY_FUNC_SEQ OF_PP_MAKE_TUPLE_SEQ("sin", Sin)

#define UNARY_FUNC_SEQ                                       \
  OF_PP_MAKE_TUPLE_SEQ("abs", Abs)                           \
  OF_PP_MAKE_TUPLE_SEQ("acos", Acos)                         \
  OF_PP_MAKE_TUPLE_SEQ("acosh", Acosh)                       \
  OF_PP_MAKE_TUPLE_SEQ("asin", Asin)                         \
  OF_PP_MAKE_TUPLE_SEQ("asinh", Asinh)                       \
  OF_PP_MAKE_TUPLE_SEQ("atan", Atan)                         \
  OF_PP_MAKE_TUPLE_SEQ("atanh", Atanh)                       \
  OF_PP_MAKE_TUPLE_SEQ("ceil", Ceil)                         \
  OF_PP_MAKE_TUPLE_SEQ("cos", Cos)                           \
  OF_PP_MAKE_TUPLE_SEQ("cosh", Cosh)                         \
  OF_PP_MAKE_TUPLE_SEQ("erf", Erf)                           \
  OF_PP_MAKE_TUPLE_SEQ("erfc", Erfc)                         \
  OF_PP_MAKE_TUPLE_SEQ("exp", Exp)                           \
  OF_PP_MAKE_TUPLE_SEQ("expm1", Expm1)                       \
  OF_PP_MAKE_TUPLE_SEQ("floor", Floor)                       \
  OF_PP_MAKE_TUPLE_SEQ("lgamma", Lgamma)                     \
  OF_PP_MAKE_TUPLE_SEQ("log", Log)                           \
  OF_PP_MAKE_TUPLE_SEQ("log1p", Log1p)                       \
  OF_PP_MAKE_TUPLE_SEQ("log_sigmoid", LogSigmoid)            \
  OF_PP_MAKE_TUPLE_SEQ("negative", Negative)                 \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal", Reciprocal)             \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal_no_nan", ReciprocalNoNan) \
  OF_PP_MAKE_TUPLE_SEQ("rint", Rint)                         \
  OF_PP_MAKE_TUPLE_SEQ("round", Round)                       \
  OF_PP_MAKE_TUPLE_SEQ("rsqrt", Rsqrt)                       \
  OF_PP_MAKE_TUPLE_SEQ("sign", Sign)                         \
  OF_PP_MAKE_TUPLE_SEQ("sinh", Sinh)                         \
  OF_PP_MAKE_TUPLE_SEQ("softplus", Softplus)                 \
  OF_PP_MAKE_TUPLE_SEQ("sqrt", Sqrt)                         \
  OF_PP_MAKE_TUPLE_SEQ("square", Square)                     \
  OF_PP_MAKE_TUPLE_SEQ("tan", Tan)

#define CAST_INT_TO_FLOAT_UNARY_FUNC_SEQ      \
  OF_PP_MAKE_TUPLE_SEQ("sigmoid_v2", Sigmoid) \
  OF_PP_MAKE_TUPLE_SEQ("tanh", Tanh)

#define UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, base)                    \
  class class_name##Functor : public base {                                          \
   public:                                                                           \
    class_name##Functor() {                                                          \
      op_ = CHECK_JUST(one::OpBuilder(op_type_name).Input("x").Output("y").Build()); \
    }                                                                                \
  };

#define INPLACEABLE_UNARY_FUNCOTRS(op_type_name, class_name) \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, InplaceableUnaryFunctor)
#define UNARY_FUNCOTRS(op_type_name, class_name) \
  UNARY_ELEMENTWISE_FUNCTOR(                     \
      op_type_name, class_name,                  \
      UnaryFunctor)  // TODO(yaochi): replace UNARY_FUNC_SEQ with INPLACEABLE_UNARY_FUNC_SEQ SEQ
#define CAST_INT_FLOAT_UNARY_FUNCOTRS(op_type_name, class_name) \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, CastIntToFloatUnaryFunctor)

OF_PP_FOR_EACH_TUPLE(INPLACEABLE_UNARY_FUNCOTRS, INPLACEABLE_UNARY_FUNC_SEQ);
OF_PP_FOR_EACH_TUPLE(UNARY_FUNCOTRS, UNARY_FUNC_SEQ);
OF_PP_FOR_EACH_TUPLE(CAST_INT_FLOAT_UNARY_FUNCOTRS, CAST_INT_TO_FLOAT_UNARY_FUNC_SEQ);

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
