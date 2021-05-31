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

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/scalar.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class ReciprocalFunctor : public UnaryFunctor {
 public:
  ReciprocalFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("reciprocal").Input("x").Output("y").Build());
  }
};

class ReciprocalNoNanFunctor : public UnaryFunctor {
 public:
  ReciprocalNoNanFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("reciprocal_no_nan").Input("x").Output("y").Build());
  }
};

class SinFunctor : public UnaryFunctor {
 public:
  SinFunctor() { op_ = CHECK_JUST(one::OpBuilder("sin").Input("x").Output("y").Build()); }
};

class CosFunctor : public UnaryFunctor {
 public:
  CosFunctor() { op_ = CHECK_JUST(one::OpBuilder("cos").Input("x").Output("y").Build()); }
};

class CoshFunctor : public UnaryFunctor {
 public:
  CoshFunctor() { op_ = CHECK_JUST(one::OpBuilder("cosh").Input("x").Output("y").Build()); }
};

class LogFunctor : public UnaryFunctor {
 public:
  LogFunctor() { op_ = CHECK_JUST(one::OpBuilder("log").Input("x").Output("y").Build()); }
};

class SqrtFunctor : public UnaryFunctor {
 public:
  SqrtFunctor() { op_ = CHECK_JUST(one::OpBuilder("sqrt").Input("x").Output("y").Build()); }
};

class RsqrtFunctor : public UnaryFunctor {
 public:
  RsqrtFunctor() { op_ = CHECK_JUST(one::OpBuilder("rsqrt").Input("x").Output("y").Build()); }
};

class SquareFunctor : public UnaryFunctor {
 public:
  SquareFunctor() { op_ = CHECK_JUST(one::OpBuilder("square").Input("x").Output("y").Build()); }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ReciprocalFunctor>("Reciprocal");
  m.add_functor<impl::ReciprocalNoNanFunctor>("ReciprocalNoNan");
  m.add_functor<impl::SinFunctor>("Sin");
  m.add_functor<impl::CosFunctor>("Cos");
  m.add_functor<impl::CoshFunctor>("Cosh");
  m.add_functor<impl::LogFunctor>("Log");
  m.add_functor<impl::SqrtFunctor>("Sqrt");
  m.add_functor<impl::RsqrtFunctor>("Rsqrt");
  m.add_functor<impl::SquareFunctor>("Square");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
