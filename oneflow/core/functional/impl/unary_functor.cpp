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
#include "oneflow/core/functional/impl/binary_functor.h"

#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/user/ops/math_unary_elementwise_seq.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

#define INPLACE_UNARY_FUNC_SEQ                                      \
  OF_PP_MAKE_TUPLE_SEQ("abs", InplaceAbs)                           \
  OF_PP_MAKE_TUPLE_SEQ("acos", InplaceAcos)                         \
  OF_PP_MAKE_TUPLE_SEQ("ceil", InplaceCeil)                         \
  OF_PP_MAKE_TUPLE_SEQ("cosh", InplaceCosh)                         \
  OF_PP_MAKE_TUPLE_SEQ("floor", InplaceFloor)                       \
  OF_PP_MAKE_TUPLE_SEQ("lgamma", InplaceLgamma)                     \
  OF_PP_MAKE_TUPLE_SEQ("log_sigmoid", InplaceLogSigmoid)            \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal_no_nan", InplaceReciprocalNoNan) \
  OF_PP_MAKE_TUPLE_SEQ("rint", InplaceRint)                         \
  OF_PP_MAKE_TUPLE_SEQ("round", InplaceRound)                       \
  OF_PP_MAKE_TUPLE_SEQ("softplus", InplaceSoftplus)

#define INPLACE_UNARY_FLOAT_FUNC_SEQ                    \
  OF_PP_MAKE_TUPLE_SEQ("acosh", InplaceAcosh)           \
  OF_PP_MAKE_TUPLE_SEQ("asin", InplaceAsin)             \
  OF_PP_MAKE_TUPLE_SEQ("asinh", InplaceAsinh)           \
  OF_PP_MAKE_TUPLE_SEQ("atan", InplaceAtan)             \
  OF_PP_MAKE_TUPLE_SEQ("atanh", InplaceAtanh)           \
  OF_PP_MAKE_TUPLE_SEQ("sin", InplaceSin)               \
  OF_PP_MAKE_TUPLE_SEQ("cos", InplaceCos)               \
  OF_PP_MAKE_TUPLE_SEQ("erf", InplaceErf)               \
  OF_PP_MAKE_TUPLE_SEQ("erfc", InplaceErfc)             \
  OF_PP_MAKE_TUPLE_SEQ("exp", InplaceExp)               \
  OF_PP_MAKE_TUPLE_SEQ("expm1", InplaceExpm1)           \
  OF_PP_MAKE_TUPLE_SEQ("log", InplaceLog)               \
  OF_PP_MAKE_TUPLE_SEQ("log1p", InplaceLog1p)           \
  OF_PP_MAKE_TUPLE_SEQ("negative", InplaceNegative)     \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal", InplaceReciprocal) \
  OF_PP_MAKE_TUPLE_SEQ("rsqrt", InplaceRsqrt)           \
  OF_PP_MAKE_TUPLE_SEQ("sigmoid_v2", InplaceSigmoid)    \
  OF_PP_MAKE_TUPLE_SEQ("sign", InplaceSign)             \
  OF_PP_MAKE_TUPLE_SEQ("sinh", InplaceSinh)             \
  OF_PP_MAKE_TUPLE_SEQ("sqrt", InplaceSqrt)             \
  OF_PP_MAKE_TUPLE_SEQ("square", InplaceSquare)         \
  OF_PP_MAKE_TUPLE_SEQ("tan", InplaceTan)               \
  OF_PP_MAKE_TUPLE_SEQ("tanh", InplaceTanh)

#define UNARY_FUNC_SEQ                                       \
  OF_PP_MAKE_TUPLE_SEQ("abs", Abs)                           \
  OF_PP_MAKE_TUPLE_SEQ("acos", Acos)                         \
  OF_PP_MAKE_TUPLE_SEQ("ceil", Ceil)                         \
  OF_PP_MAKE_TUPLE_SEQ("cosh", Cosh)                         \
  OF_PP_MAKE_TUPLE_SEQ("floor", Floor)                       \
  OF_PP_MAKE_TUPLE_SEQ("lgamma", Lgamma)                     \
  OF_PP_MAKE_TUPLE_SEQ("log_sigmoid", LogSigmoid)            \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal_no_nan", ReciprocalNoNan) \
  OF_PP_MAKE_TUPLE_SEQ("rint", Rint)                         \
  OF_PP_MAKE_TUPLE_SEQ("round", Round)                       \
  OF_PP_MAKE_TUPLE_SEQ("softplus", Softplus)

#define FLOAT_UNARY_FUNC_SEQ                     \
  OF_PP_MAKE_TUPLE_SEQ("acosh", Acosh)           \
  OF_PP_MAKE_TUPLE_SEQ("asin", Asin)             \
  OF_PP_MAKE_TUPLE_SEQ("asinh", Asinh)           \
  OF_PP_MAKE_TUPLE_SEQ("atan", Atan)             \
  OF_PP_MAKE_TUPLE_SEQ("atanh", Atanh)           \
  OF_PP_MAKE_TUPLE_SEQ("sin", Sin)               \
  OF_PP_MAKE_TUPLE_SEQ("cos", Cos)               \
  OF_PP_MAKE_TUPLE_SEQ("erf", Erf)               \
  OF_PP_MAKE_TUPLE_SEQ("erfc", Erfc)             \
  OF_PP_MAKE_TUPLE_SEQ("exp", Exp)               \
  OF_PP_MAKE_TUPLE_SEQ("expm1", Expm1)           \
  OF_PP_MAKE_TUPLE_SEQ("log", Log)               \
  OF_PP_MAKE_TUPLE_SEQ("log1p", Log1p)           \
  OF_PP_MAKE_TUPLE_SEQ("negative", Negative)     \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal", Reciprocal) \
  OF_PP_MAKE_TUPLE_SEQ("rsqrt", Rsqrt)           \
  OF_PP_MAKE_TUPLE_SEQ("sigmoid_v2", Sigmoid)    \
  OF_PP_MAKE_TUPLE_SEQ("sign", Sign)             \
  OF_PP_MAKE_TUPLE_SEQ("sinh", Sinh)             \
  OF_PP_MAKE_TUPLE_SEQ("sqrt", Sqrt)             \
  OF_PP_MAKE_TUPLE_SEQ("square", Square)         \
  OF_PP_MAKE_TUPLE_SEQ("tan", Tan)               \
  OF_PP_MAKE_TUPLE_SEQ("tanh", Tanh)

#define LOGICAL_FLOAT_UNARY_FUNC_SEQ OF_PP_MAKE_TUPLE_SEQ("logical_not", LogicalNot)

#define UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, base)                    \
  class class_name##Functor : public base {                                          \
   public:                                                                           \
    class_name##Functor() {                                                          \
      op_ = CHECK_JUST(one::OpBuilder(op_type_name).Input("x").Output("y").Build()); \
    }                                                                                \
  };

#define UNARY_ELEMENTWISE_GRAD_FUNCTOR(op_type_name, class_name, base)          \
  class class_name##GradFunctor : public base {                                 \
   public:                                                                      \
    class_name##GradFunctor() {                                                 \
      op_ = CHECK_JUST(one::OpBuilder(std::string("") + op_type_name + "_grad") \
                           .Input("x")                                          \
                           .Input("dy")                                         \
                           .Output("dx")                                        \
                           .Build());                                           \
    }                                                                           \
  };

#define INPLACE_UNARY_FUNCOTRS(op_type_name, class_name) \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, InplaceUnaryFunctor)
#define INPLACE_FLOAT_UNARY_FUNCOTRS(op_type_name, class_name) \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, InplaceFloatUnaryFunctor)
#define LOGICAL_FLOAT_UNARY_FUNCTORS(op_type_name, class_name) \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, FloatUnaryFunctor)
#define UNARY_FUNCOTRS(op_type_name, class_name)                    \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, UnaryFunctor) \
  UNARY_ELEMENTWISE_GRAD_FUNCTOR(op_type_name, class_name, BinaryFunctor)
#define FLOAT_UNARY_FUNCOTRS(op_type_name, class_name)                   \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, FloatUnaryFunctor) \
  UNARY_ELEMENTWISE_GRAD_FUNCTOR(op_type_name, class_name, BinaryFunctor)

OF_PP_FOR_EACH_TUPLE(INPLACE_UNARY_FUNCOTRS, INPLACE_UNARY_FUNC_SEQ)
OF_PP_FOR_EACH_TUPLE(INPLACE_FLOAT_UNARY_FUNCOTRS, INPLACE_UNARY_FLOAT_FUNC_SEQ);
OF_PP_FOR_EACH_TUPLE(UNARY_FUNCOTRS, UNARY_FUNC_SEQ);
OF_PP_FOR_EACH_TUPLE(FLOAT_UNARY_FUNCOTRS, FLOAT_UNARY_FUNC_SEQ);
OF_PP_FOR_EACH_TUPLE(LOGICAL_FLOAT_UNARY_FUNCTORS, LOGICAL_FLOAT_UNARY_FUNC_SEQ);

}  // namespace impl

using namespace impl;
#define ADD_UNARY_FUNCTOR(class_name, functor_name) \
  m.add_functor<class_name##Functor>(functor_name); \
  m.add_functor<class_name##GradFunctor>(std::string("") + functor_name + "Grad");

ONEFLOW_FUNCTION_LIBRARY(m) {
  ADD_UNARY_FUNCTOR(Abs, "Abs");
  ADD_UNARY_FUNCTOR(Acos, "Acos");
  ADD_UNARY_FUNCTOR(Acosh, "Acosh");
  ADD_UNARY_FUNCTOR(Asin, "Asin");
  ADD_UNARY_FUNCTOR(Asinh, "Asinh");
  ADD_UNARY_FUNCTOR(Atan, "Atan");
  ADD_UNARY_FUNCTOR(Atanh, "Atanh");
  ADD_UNARY_FUNCTOR(Ceil, "Ceil");
  ADD_UNARY_FUNCTOR(Cos, "Cos");
  ADD_UNARY_FUNCTOR(Cosh, "Cosh");
  ADD_UNARY_FUNCTOR(Erf, "Erf");
  ADD_UNARY_FUNCTOR(Erfc, "Erfc");
  ADD_UNARY_FUNCTOR(Exp, "Exp");
  ADD_UNARY_FUNCTOR(Expm1, "Expm1");
  ADD_UNARY_FUNCTOR(Floor, "Floor");
  ADD_UNARY_FUNCTOR(Lgamma, "Lgamma");
  ADD_UNARY_FUNCTOR(Log, "Log");
  ADD_UNARY_FUNCTOR(Log1p, "Log1p");
  ADD_UNARY_FUNCTOR(LogSigmoid, "LogSigmoid");
  ADD_UNARY_FUNCTOR(Negative, "Negative");
  ADD_UNARY_FUNCTOR(Reciprocal, "Reciprocal");
  ADD_UNARY_FUNCTOR(ReciprocalNoNan, "ReciprocalNoNan");
  ADD_UNARY_FUNCTOR(Rint, "Rint");
  ADD_UNARY_FUNCTOR(Round, "Round");
  ADD_UNARY_FUNCTOR(Rsqrt, "Rsqrt");
  ADD_UNARY_FUNCTOR(Sigmoid, "Sigmoid");
  ADD_UNARY_FUNCTOR(Sign, "Sign");
  ADD_UNARY_FUNCTOR(Sin, "Sin");
  ADD_UNARY_FUNCTOR(Sinh, "Sinh");
  ADD_UNARY_FUNCTOR(Softplus, "Softplus");
  ADD_UNARY_FUNCTOR(Sqrt, "Sqrt");
  ADD_UNARY_FUNCTOR(Square, "Square");
  ADD_UNARY_FUNCTOR(Tan, "Tan");
  ADD_UNARY_FUNCTOR(Tanh, "Tanh");
  m.add_functor<LogicalNotFunctor>("LogicalNot");
  // inplace version of the functors above
  m.add_functor<impl::InplaceAbsFunctor>("Abs_");
  m.add_functor<impl::InplaceAcosFunctor>("Acos_");
  m.add_functor<impl::InplaceAcoshFunctor>("Acosh_");
  m.add_functor<impl::InplaceAsinFunctor>("Asin_");
  m.add_functor<impl::InplaceAsinhFunctor>("Asinh_");
  m.add_functor<impl::InplaceAtanFunctor>("Atan_");
  m.add_functor<impl::InplaceAtanhFunctor>("Atanh_");
  m.add_functor<impl::InplaceCeilFunctor>("Ceil_");
  m.add_functor<impl::InplaceCosFunctor>("Cos_");
  m.add_functor<impl::InplaceCoshFunctor>("Cosh_");
  m.add_functor<impl::InplaceErfFunctor>("Erf_");
  m.add_functor<impl::InplaceErfcFunctor>("Erfc_");
  m.add_functor<impl::InplaceExpFunctor>("Exp_");
  m.add_functor<impl::InplaceExpm1Functor>("Expm1_");
  m.add_functor<impl::InplaceFloorFunctor>("Floor_");
  m.add_functor<impl::InplaceLgammaFunctor>("Lgamma_");
  m.add_functor<impl::InplaceLogFunctor>("Log_");
  m.add_functor<impl::InplaceLog1pFunctor>("Log1p_");
  m.add_functor<impl::InplaceLogSigmoidFunctor>("LogSigmoid_");
  m.add_functor<impl::InplaceNegativeFunctor>("Negative_");
  m.add_functor<impl::InplaceReciprocalFunctor>("Reciprocal_");
  m.add_functor<impl::InplaceReciprocalNoNanFunctor>("ReciprocalNoNan_");
  m.add_functor<impl::InplaceRintFunctor>("Rint_");
  m.add_functor<impl::InplaceRoundFunctor>("Round_");
  m.add_functor<impl::InplaceRsqrtFunctor>("Rsqrt_");
  m.add_functor<impl::InplaceSigmoidFunctor>("Sigmoid_");
  m.add_functor<impl::InplaceSignFunctor>("Sign_");
  m.add_functor<impl::InplaceSinFunctor>("Sin_");
  m.add_functor<impl::InplaceSinhFunctor>("Sinh_");
  m.add_functor<impl::InplaceSoftplusFunctor>("Softplus_");
  m.add_functor<impl::InplaceSqrtFunctor>("Sqrt_");
  m.add_functor<impl::InplaceSquareFunctor>("Square_");
  m.add_functor<impl::InplaceTanFunctor>("Tan_");
  m.add_functor<impl::InplaceTanhFunctor>("Tanh_");
};

#undef ADD_UNARY_FUNCTOR

}  // namespace functional
}  // namespace one
}  // namespace oneflow