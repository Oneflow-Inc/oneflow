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

#define INPLACE_UNARY_FLOAT_FUNC_SEQ          \
  OF_PP_MAKE_TUPLE_SEQ("sin", InplaceSin)     \
  OF_PP_MAKE_TUPLE_SEQ("floor", InplaceFloor) \
  OF_PP_MAKE_TUPLE_SEQ("ceil", InplaceCeil)   \
  OF_PP_MAKE_TUPLE_SEQ("round", InplaceRound)

#define UNARY_PRIMITIVE_FUNC_BWD_WITH_DY_X_SEQ    \
  OF_PP_MAKE_TUPLE_SEQ("abs", Abs)                \
  OF_PP_MAKE_TUPLE_SEQ("acos", Acos)              \
  OF_PP_MAKE_TUPLE_SEQ("cosh", Cosh)              \
  OF_PP_MAKE_TUPLE_SEQ("lgamma", Lgamma)          \
  OF_PP_MAKE_TUPLE_SEQ("log_sigmoid", LogSigmoid) \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal_no_nan", ReciprocalNoNan)

#define FLOAT_UNARY_PRIMITIVE_FUNC_BWD_WITH_DY_X_SEQ \
  OF_PP_MAKE_TUPLE_SEQ("acosh", Acosh)               \
  OF_PP_MAKE_TUPLE_SEQ("asin", Asin)                 \
  OF_PP_MAKE_TUPLE_SEQ("asinh", Asinh)               \
  OF_PP_MAKE_TUPLE_SEQ("atan", Atan)                 \
  OF_PP_MAKE_TUPLE_SEQ("atanh", Atanh)               \
  OF_PP_MAKE_TUPLE_SEQ("sin", Sin)                   \
  OF_PP_MAKE_TUPLE_SEQ("cos", Cos)                   \
  OF_PP_MAKE_TUPLE_SEQ("erf", Erf)                   \
  OF_PP_MAKE_TUPLE_SEQ("erfc", Erfc)                 \
  OF_PP_MAKE_TUPLE_SEQ("exp", Exp)                   \
  OF_PP_MAKE_TUPLE_SEQ("exp2", Exp2)                 \
  OF_PP_MAKE_TUPLE_SEQ("expm1", Expm1)               \
  OF_PP_MAKE_TUPLE_SEQ("log", Log)                   \
  OF_PP_MAKE_TUPLE_SEQ("log2", Log2)                 \
  OF_PP_MAKE_TUPLE_SEQ("log10", Log10)               \
  OF_PP_MAKE_TUPLE_SEQ("log1p", Log1p)               \
  OF_PP_MAKE_TUPLE_SEQ("reciprocal", Reciprocal)     \
  OF_PP_MAKE_TUPLE_SEQ("rsqrt", Rsqrt)               \
  OF_PP_MAKE_TUPLE_SEQ("sinh", Sinh)                 \
  OF_PP_MAKE_TUPLE_SEQ("sqrt", Sqrt)                 \
  OF_PP_MAKE_TUPLE_SEQ("square", Square)             \
  OF_PP_MAKE_TUPLE_SEQ("tan", Tan)                   \
  OF_PP_MAKE_TUPLE_SEQ("tanh", Tanh)

#define FLOAT_UNARY_PRIMITIVE_FUNC_BWD_WITH_DY_Y_SEQ OF_PP_MAKE_TUPLE_SEQ("sigmoid", Sigmoid)

#define UNARY_FUNC_BWD_WITH_FILL_SEQ   \
  OF_PP_MAKE_TUPLE_SEQ("rint", Rint)   \
  OF_PP_MAKE_TUPLE_SEQ("round", Round) \
  OF_PP_MAKE_TUPLE_SEQ("floor", Floor) \
  OF_PP_MAKE_TUPLE_SEQ("ceil", Ceil)

#define FLOAT_UNARY_FUNC_BWD_WITH_FILL_SEQ \
  OF_PP_MAKE_TUPLE_SEQ("sign", Sign)       \
  OF_PP_MAKE_TUPLE_SEQ("not_equal_zero", NotEqualZero)

#define LOGICAL_FLOAT_UNARY_FUNC_SEQ OF_PP_MAKE_TUPLE_SEQ("logical_not", LogicalNot)

#define UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, base)                    \
  class class_name##Functor : public base {                                          \
   public:                                                                           \
    class_name##Functor() {                                                          \
      op_ = CHECK_JUST(one::OpBuilder(op_type_name).Input("x").Output("y").Build()); \
    }                                                                                \
  };

#define UNARY_ELEMENTWISE_BWD_WITH_DY_X_FUNCTOR(op_type_name, class_name, base) \
  class class_name##WithDyXGradFunctor : public base {                          \
   public:                                                                      \
    class_name##WithDyXGradFunctor() {                                          \
      op_ = CHECK_JUST(one::OpBuilder(std::string("") + op_type_name + "_grad") \
                           .Input("x")                                          \
                           .Input("dy")                                         \
                           .Output("dx")                                        \
                           .Build());                                           \
    }                                                                           \
  };

#define UNARY_ELEMENTWISE_BWD_WITH_DY_Y_FUNCTOR(op_type_name, class_name, base) \
  class class_name##WithDyYGradFunctor : public base {                          \
   public:                                                                      \
    class_name##WithDyYGradFunctor() {                                          \
      op_ = CHECK_JUST(one::OpBuilder(std::string("") + op_type_name + "_grad") \
                           .Input("y")                                          \
                           .Input("dy")                                         \
                           .Output("dx")                                        \
                           .Build());                                           \
    }                                                                           \
  };

#define INPLACE_UNARY_FUNCTORS(op_type_name, class_name) \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, InplaceUnaryFunctor)
#define INPLACE_FLOAT_UNARY_FUNCTORS(op_type_name, class_name) \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, InplaceFloatUnaryFunctor)
#define LOGICAL_FLOAT_UNARY_FUNCTORS(op_type_name, class_name) \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, FloatUnaryFunctor)
#define UNARY_FUNCTORS(op_type_name, class_name)                    \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, UnaryFunctor) \
  UNARY_ELEMENTWISE_BWD_WITH_DY_X_FUNCTOR(op_type_name, class_name, BinaryFunctor)
#define FLOAT_UNARY_FUNCTORS(op_type_name, class_name)                   \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, FloatUnaryFunctor) \
  UNARY_ELEMENTWISE_BWD_WITH_DY_X_FUNCTOR(op_type_name, class_name, BinaryFunctor)

#define UNARY_BWD_WITH_DY_X_FUNCTORS(op_type_name, class_name)      \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, UnaryFunctor) \
  UNARY_ELEMENTWISE_BWD_WITH_DY_X_FUNCTOR(op_type_name, class_name, BinaryFunctor)

#define FLOAT_UNARY_BWD_WITH_DY_X_FUNCTORS(op_type_name, class_name)     \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, FloatUnaryFunctor) \
  UNARY_ELEMENTWISE_BWD_WITH_DY_X_FUNCTOR(op_type_name, class_name, BinaryFunctor)

#define FLOAT_UNARY_WITH_DY_Y_FUNCTORS(op_type_name, class_name)         \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, FloatUnaryFunctor) \
  UNARY_ELEMENTWISE_BWD_WITH_DY_Y_FUNCTOR(op_type_name, class_name, BinaryFunctor)

#define FLOAT_UNARY_BWD_WITH_FILL_FUNCTORS(op_type_name, class_name) \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, FloatUnaryFunctor)

#define UNARY_BWD_WITH_FILL_FUNCTORS(op_type_name, class_name) \
  UNARY_ELEMENTWISE_FUNCTOR(op_type_name, class_name, UnaryFunctor)

OF_PP_FOR_EACH_TUPLE(INPLACE_FLOAT_UNARY_FUNCTORS, INPLACE_UNARY_FLOAT_FUNC_SEQ);
OF_PP_FOR_EACH_TUPLE(LOGICAL_FLOAT_UNARY_FUNCTORS, LOGICAL_FLOAT_UNARY_FUNC_SEQ);

OF_PP_FOR_EACH_TUPLE(UNARY_BWD_WITH_DY_X_FUNCTORS, UNARY_PRIMITIVE_FUNC_BWD_WITH_DY_X_SEQ);
OF_PP_FOR_EACH_TUPLE(FLOAT_UNARY_BWD_WITH_DY_X_FUNCTORS,
                     FLOAT_UNARY_PRIMITIVE_FUNC_BWD_WITH_DY_X_SEQ);

OF_PP_FOR_EACH_TUPLE(FLOAT_UNARY_WITH_DY_Y_FUNCTORS, FLOAT_UNARY_PRIMITIVE_FUNC_BWD_WITH_DY_Y_SEQ);
OF_PP_FOR_EACH_TUPLE(UNARY_BWD_WITH_FILL_FUNCTORS, UNARY_FUNC_BWD_WITH_FILL_SEQ);
OF_PP_FOR_EACH_TUPLE(FLOAT_UNARY_BWD_WITH_FILL_FUNCTORS, FLOAT_UNARY_FUNC_BWD_WITH_FILL_SEQ);

UNARY_ELEMENTWISE_FUNCTOR("negative", Negative, FloatUnaryFunctor)
UNARY_ELEMENTWISE_FUNCTOR("bitwise_not", BitwiseNot, UnaryFunctor)

}  // namespace impl

using namespace impl;
#define ADD_UNARY_FUNCTOR_WITH_DY_X(class_name, functor_name) \
  m.add_functor<class_name##Functor>(functor_name);           \
  m.add_functor<class_name##WithDyXGradFunctor>(std::string("") + functor_name + "Grad");

#define ADD_UNARY_FUNCTOR_WITH_DY_Y(class_name, functor_name) \
  m.add_functor<class_name##Functor>(functor_name);           \
  m.add_functor<class_name##WithDyYGradFunctor>(std::string("") + functor_name + "Grad");

ONEFLOW_FUNCTION_LIBRARY(m) {
  ADD_UNARY_FUNCTOR_WITH_DY_X(Abs, "Abs");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Acos, "Acos");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Acosh, "Acosh");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Asin, "Asin");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Asinh, "Asinh");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Atan, "Atan");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Atanh, "Atanh");
  m.add_functor<CeilFunctor>("Ceil");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Cos, "Cos");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Cosh, "Cosh");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Erf, "Erf");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Erfc, "Erfc");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Exp, "Exp");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Exp2, "Exp2");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Expm1, "Expm1");
  m.add_functor<FloorFunctor>("Floor");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Lgamma, "Lgamma");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Log, "Log");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Log2, "Log2");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Log10, "Log10");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Log1p, "Log1p");
  ADD_UNARY_FUNCTOR_WITH_DY_X(LogSigmoid, "LogSigmoid");
  m.add_functor<NegativeFunctor>("Negative");
  m.add_functor<BitwiseNotFunctor>("BitwiseNot");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Reciprocal, "Reciprocal");
  ADD_UNARY_FUNCTOR_WITH_DY_X(ReciprocalNoNan, "ReciprocalNoNan");
  m.add_functor<RintFunctor>("Rint");
  m.add_functor<RoundFunctor>("Round");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Rsqrt, "Rsqrt");
  ADD_UNARY_FUNCTOR_WITH_DY_Y(Sigmoid, "Sigmoid");
  m.add_functor<SignFunctor>("Sign");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Sin, "Sin");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Sinh, "Sinh");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Sqrt, "Sqrt");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Square, "Square");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Tan, "Tan");
  ADD_UNARY_FUNCTOR_WITH_DY_X(Tanh, "Tanh");
  m.add_functor<NotEqualZeroFunctor>("NotEqualZero");
  m.add_functor<LogicalNotFunctor>("LogicalNot");
  m.add_functor<InplaceSinFunctor>("Sin_");
  m.add_functor<InplaceFloorFunctor>("Floor_");
  m.add_functor<InplaceCeilFunctor>("Ceil_");
  m.add_functor<InplaceRoundFunctor>("Round_");
};

#undef ADD_UNARY_FUNCTOR_WITH_DY_X
#undef ADD_UNARY_FUNCTOR_WITH_DY_Y

}  // namespace functional
}  // namespace one
}  // namespace oneflow