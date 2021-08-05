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

class AddFunctor : public InplaceableBinaryFunctor {
 public:
  AddFunctor() { op_ = CHECK_JUST(one::OpBuilder("add_n").Input("in", 2).Output("out").Build()); }
};

class MultiplyFunctor : public BinaryFunctor {
 public:
  MultiplyFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("multiply").Input("x").Input("y").Output("out").Build());
  }
};

class PowFunctor : public BinaryFunctor {
 public:
  PowFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("pow").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastAddFunctor : public BinaryFunctor {
 public:
  BroadcastAddFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_add").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastSubFunctor : public BinaryFunctor {
 public:
  BroadcastSubFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_sub").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastMulFunctor : public BinaryFunctor {
 public:
  BroadcastMulFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_mul").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastDivFunctor : public BinaryFunctor {
 public:
  BroadcastDivFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_div").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastFModFunctor : public BinaryFunctor {
 public:
  BroadcastFModFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_fmod").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastEqualFunctor : public BinaryFunctor {
 public:
  BroadcastEqualFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_equal").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastNotEqualFunctor : public BinaryFunctor {
 public:
  BroadcastNotEqualFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("broadcast_not_equal").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastGreaterFunctor : public BinaryFunctor {
 public:
  BroadcastGreaterFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_greater").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastGreaterEqualFunctor : public BinaryFunctor {
 public:
  BroadcastGreaterEqualFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("broadcast_greater_equal").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastLessFunctor : public BinaryFunctor {
 public:
  BroadcastLessFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_less").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastLessEqualFunctor : public BinaryFunctor {
 public:
  BroadcastLessEqualFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("broadcast_less_equal").Input("x").Input("y").Output("z").Build());
  }
};

class ScalarAddByTensorFunctor : public InplaceableBinaryFunctor {
 public:
  ScalarAddByTensorFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("scalar_add_by_tensor").Input("x").Input("scalar").Output("y").Build());
  }
};

class ScalarSubByTensorFunctor : public BinaryFunctor {
 public:
  ScalarSubByTensorFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("scalar_sub_by_tensor").Input("x").Input("scalar").Output("y").Build());
  }
};

class ScalarMulByTensorFunctor : public BinaryFunctor {
 public:
  ScalarMulByTensorFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("scalar_mul_by_tensor").Input("x").Input("scalar").Output("y").Build());
  }
};

class ScalarDivByTensorFunctor : public BinaryFunctor {
 public:
  ScalarDivByTensorFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("scalar_div_by_tensor").Input("x").Input("scalar").Output("y").Build());
  }
};

class BroadcastMinimumFunctor : public BinaryFunctor {
 public:
  BroadcastMinimumFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_minimum").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastMaximumFunctor : public BinaryFunctor {
 public:
  BroadcastMaximumFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_maximum").Input("x").Input("y").Output("z").Build());
  }
};

class ElementwiseMinimumFunctor : public BinaryFunctor {
 public:
  ElementwiseMinimumFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("elementwise_minimum").Input("x").Input("y").Output("z").Build());
  }
};

class ElementwiseMaximumFunctor : public BinaryFunctor {
 public:
  ElementwiseMaximumFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("elementwise_maximum").Input("x").Input("y").Output("z").Build());
  }
};

class ReshapeLikeFunctor : public BinaryFunctor {
 public:
  ReshapeLikeFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("reshape_like").Input("in").Input("like").Output("out").Build());
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::AddFunctor>("Add");
  m.add_functor<impl::MultiplyFunctor>("Multiply");
  m.add_functor<impl::PowFunctor>("Pow");
  m.add_functor<impl::BroadcastAddFunctor>("BroadcastAdd");
  m.add_functor<impl::BroadcastSubFunctor>("BroadcastSub");
  m.add_functor<impl::BroadcastMulFunctor>("BroadcastMul");
  m.add_functor<impl::BroadcastDivFunctor>("BroadcastDiv");
  m.add_functor<impl::BroadcastMinimumFunctor>("BroadcastMin");
  m.add_functor<impl::BroadcastMaximumFunctor>("BroadcastMax");
  m.add_functor<impl::BroadcastEqualFunctor>("BroadcastEqual");
  m.add_functor<impl::BroadcastNotEqualFunctor>("BroadcastNotEqual");
  m.add_functor<impl::BroadcastGreaterFunctor>("BroadcastGreater");
  m.add_functor<impl::BroadcastGreaterEqualFunctor>("BroadcastGreaterEqual");
  m.add_functor<impl::BroadcastLessFunctor>("BroadcastLess");
  m.add_functor<impl::BroadcastLessEqualFunctor>("BroadcastLessEqual");
  m.add_functor<impl::ScalarAddByTensorFunctor>("ScalarAddByTensor");
  m.add_functor<impl::ScalarSubByTensorFunctor>("ScalarSubByTensor");
  m.add_functor<impl::ScalarMulByTensorFunctor>("ScalarMulByTensor");
  m.add_functor<impl::ScalarDivByTensorFunctor>("ScalarDivByTensor");
  m.add_functor<impl::ElementwiseMinimumFunctor>("ElementwiseMin");
  m.add_functor<impl::ElementwiseMaximumFunctor>("ElementwiseMax");
  m.add_functor<impl::BroadcastFModFunctor>("BroadcastFMod");
  m.add_functor<impl::ReshapeLikeFunctor>("ReshapeLike");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
