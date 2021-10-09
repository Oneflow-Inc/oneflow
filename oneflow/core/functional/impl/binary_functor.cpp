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

#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class AddFunctor {
 public:
  AddFunctor() {
    add_op_ = CHECK_JUST(one::OpBuilder("add_n").Input("in", 2).Output("out").Build());
    broadcast_add_op_ =
        CHECK_JUST(one::OpBuilder("broadcast_add").Input("x").Input("y").Output("z").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y, bool inplace) const {
    bool x_static_zeros = IsStaticZerosTensor(x);
    if (x_static_zeros || IsStaticZerosTensor(y)) {
      CHECK_OR_RETURN(JUST(x->device()) == JUST(y->device()));
      CHECK_OR_RETURN(*x->shape() == *y->shape());
      CHECK_OR_RETURN(x->dtype() == y->dtype());
      if (x_static_zeros) { return y; }
      return x;
    }

    const OpExpr* op = nullptr;

    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({x, y}).Apply());
    TensorTuple input_vec = JUST(tensor_processor.GetInputs());
    const std::shared_ptr<one::Tensor>& x_cast = input_vec[0];
    const std::shared_ptr<one::Tensor>& y_cast = input_vec[1];

    if (*x_cast->shape() == *y_cast->shape()) {
      op = add_op_.get();
    } else {
      op = broadcast_add_op_.get();
    }
    if (inplace) {
      JUST(CheckInplaceCastValid(x, x_cast));
      JUST(CheckInplaceValid(x));
      JUST(CheckShapeCanExpandTo(*y_cast->shape(), *x_cast->shape()));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = x_cast;
      JUST(OpInterpUtil::Dispatch(*op, input_vec, outputs.get()));
      return outputs->at(0);
    }
    return OpInterpUtil::Dispatch<Tensor>(*op, input_vec);
  }

 private:
  std::shared_ptr<OpExpr> add_op_;
  std::shared_ptr<OpExpr> broadcast_add_op_;
};

class BroadcastPowFunctor : public BinaryFloatFunctor {
 public:
  BroadcastPowFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_pow").Input("x").Input("y").Output("z").Build());
  }
};

class SubFunctor : public BinaryFunctor {
 public:
  SubFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_sub").Input("x").Input("y").Output("z").Build());
  }
};

class MulFunctor {
 public:
  MulFunctor() {
    mul_op_ = CHECK_JUST(one::OpBuilder("multiply").Input("x").Input("y").Output("out").Build());
    broadcast_mul_op_ =
        CHECK_JUST(one::OpBuilder("broadcast_mul").Input("x").Input("y").Output("z").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({x, y}).Apply());
    TensorTuple input_vec = JUST(tensor_processor.GetInputs());

    if (*x->shape() == *y->shape()) { return OpInterpUtil::Dispatch<Tensor>(*mul_op_, input_vec); }
    return OpInterpUtil::Dispatch<Tensor>(*broadcast_mul_op_, input_vec);
  }

 private:
  std::shared_ptr<OpExpr> mul_op_;
  std::shared_ptr<OpExpr> broadcast_mul_op_;
};

class DivFunctor : public BinaryFloatFunctor {
 public:
  DivFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_div").Input("x").Input("y").Output("z").Build());
  }
};

class Atan2Functor : public BinaryFloatFunctor {
 public:
  Atan2Functor() {
    op_ = CHECK_JUST(one::OpBuilder("atan2").Input("x").Input("y").Output("z").Build());
  }
};

class PowFunctor : public BinaryFloatFunctor {
 public:
  PowFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("pow").Input("x").Input("y").Output("z").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    if (*x->shape() != *y->shape()) { return BroadcastPow(x, y); }
    return BinaryFloatFunctor::operator()(x, y);
  }
};

class FloorDivFunctor : public BinaryFunctor {
 public:
  FloorDivFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("floordiv").Input("x").Input("y").Output("z").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    return BinaryFunctor::operator()(x, y);
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

class BroadcastLogicalAndFunctor : public BinaryFunctor {
 public:
  BroadcastLogicalAndFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("broadcast_logical_and").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastLogicalOrFunctor : public BinaryFunctor {
 public:
  BroadcastLogicalOrFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("broadcast_logical_or").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastLogicalXorFunctor : public BinaryFunctor {
 public:
  BroadcastLogicalXorFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("broadcast_logical_xor").Input("x").Input("y").Output("z").Build());
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
  m.add_functor<impl::Atan2Functor>("Atan2");
  m.add_functor<impl::SubFunctor>("Sub");
  m.add_functor<impl::MulFunctor>("Mul");
  m.add_functor<impl::DivFunctor>("Div");
  m.add_functor<impl::PowFunctor>("Pow");
  m.add_functor<impl::BroadcastPowFunctor>("BroadcastPow");
  m.add_functor<impl::BroadcastEqualFunctor>("BroadcastEqual");
  m.add_functor<impl::BroadcastNotEqualFunctor>("BroadcastNotEqual");
  m.add_functor<impl::BroadcastGreaterFunctor>("BroadcastGreater");
  m.add_functor<impl::BroadcastGreaterEqualFunctor>("BroadcastGreaterEqual");
  m.add_functor<impl::BroadcastLogicalAndFunctor>("BroadcastLogicalAnd");
  m.add_functor<impl::BroadcastLogicalOrFunctor>("BroadcastLogicalOr");
  m.add_functor<impl::BroadcastLogicalXorFunctor>("BroadcastLogicalXor");
  m.add_functor<impl::BroadcastLessFunctor>("BroadcastLess");
  m.add_functor<impl::BroadcastLessEqualFunctor>("BroadcastLessEqual");
  m.add_functor<impl::ScalarAddByTensorFunctor>("ScalarAddByTensor");
  m.add_functor<impl::ScalarSubByTensorFunctor>("ScalarSubByTensor");
  m.add_functor<impl::ScalarMulByTensorFunctor>("ScalarMulByTensor");
  m.add_functor<impl::ScalarDivByTensorFunctor>("ScalarDivByTensor");
  m.add_functor<impl::BroadcastFModFunctor>("BroadcastFMod");
  m.add_functor<impl::ReshapeLikeFunctor>("ReshapeLike");
  m.add_functor<impl::FloorDivFunctor>("FloorDiv");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
