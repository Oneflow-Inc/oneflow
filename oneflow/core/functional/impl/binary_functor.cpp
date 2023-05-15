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

#include "oneflow/core/common/error.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/sequence_function.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

namespace {

bool IsCPUScalarTensor(const std::shared_ptr<Tensor>& tensor) {
  return tensor->shape()->NumAxes() == 0
         && TensorDeviceToString(tensor).find("cpu") != std::string::npos;
}

}  // namespace

std::string TensorDeviceToString(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_global()) { return CHECK_JUST(tensor->parallel_desc())->device_tag(); }
  return CHECK_JUST(tensor->device())->ToString();
}

Maybe<void> CastDeviceForCPUScalarTensor(std::shared_ptr<Tensor>& tensor,
                                         std::shared_ptr<Tensor>& other, bool inplace) {
  if (TensorDeviceToString(tensor) != TensorDeviceToString(other)) {
    if (IsCPUScalarTensor(other)) {
      other = JUST(functional::To(other, TensorDeviceToString(tensor)));
    } else if (!inplace && IsCPUScalarTensor(tensor)) {
      tensor = JUST(functional::To(tensor, TensorDeviceToString(other)));
    }
  }
  return Maybe<void>::Ok();
}

class AddFunctor {
 public:
  AddFunctor() {
    add_op_ = CHECK_JUST(one::OpBuilder("add_n").Input("in", 2).Output("out").Build());
    broadcast_add_op_ =
        CHECK_JUST(one::OpBuilder("broadcast_add").Input("x").Input("y").Output("z").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& other, const Scalar& alpha,
                           bool inplace) const {
    auto input_tensor = input;
    if (IsIntegralDataType(input_tensor->dtype()->data_type())
        && IsIntegralDataType(other->dtype()->data_type()) && alpha.IsFloatingPoint()) {
      return Error::RuntimeError()
             << "For integral input tensors, argument alpha must not be a floating point number.";
    }

    bool input_static_zeros = IsStaticZerosTensor(input_tensor);
    if (input_static_zeros || IsStaticZerosTensor(other)) {
      CHECK_OR_RETURN(JUST(input_tensor->device()) == JUST(other->device()))
          << Error::RuntimeError()
          << "Expected all tensors to be on the same device, but found at least two devices, "
          << JUST(input_tensor->device())->ToString() << " and "
          << JUST(other->device())->ToString() << "!";
      CHECK_OR_RETURN(*input_tensor->shape() == *other->shape())
          << Error::RuntimeError() << "The size of tensor a " << input_tensor->shape()->ToString()
          << " must match the size of tensor b " << other->shape();
      if (input_static_zeros) {
        if ((alpha.IsIntegral() && alpha.Value<int64_t>() == 1)
            || (alpha.IsFloatingPoint()
                && std::fabs(alpha.Value<double>() - 1.0)
                       < std::numeric_limits<double>::epsilon())) {
          return other;
        } else {
          return JUST(functional::ScalarMul(alpha, other));
        }
      }
      return input_tensor;
    }

    const OpExpr* op = nullptr;
    Optional<Symbol<DType>> promote_dtype;
    if (inplace) { promote_dtype = input_tensor->dtype(); }

    TensorProcessor tensor_processor;
    if ((alpha.IsIntegral() && alpha.Value<int64_t>() == 1)
        || (alpha.IsFloatingPoint()
            && std::fabs(alpha.Value<double>() - 1.0) < std::numeric_limits<double>::epsilon())) {
      JUST(tensor_processor.PromoteInputsToCommonDtype(true, promote_dtype)
               .AddInputs({input_tensor, other})
               .Apply());
    } else {
      JUST(tensor_processor.PromoteInputsToCommonDtype(true, promote_dtype)
               .AddInputs({input_tensor, JUST(functional::ScalarMul(alpha, other))})
               .Apply());
    }
    TensorTuple input_vec = JUST(tensor_processor.GetInputs());
    const std::shared_ptr<one::Tensor>& input_cast = input_vec[0];
    const std::shared_ptr<one::Tensor>& other_cast = input_vec[1];
    JUST(CastDeviceForCPUScalarTensor(input_vec[0], input_vec[1], inplace));

    if (*input_cast->shape() == *other_cast->shape()) {
      op = add_op_.get();
    } else {
      op = broadcast_add_op_.get();
    }
    if (inplace) {
      JUST(CheckInplaceCastValid(input_tensor, input_cast));
      JUST(CheckInplaceValid(input_tensor));
      JUST(CheckInplaceShapeCanExpandTo(*other_cast->shape(), *input_cast->shape()));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = input_cast;
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

class SubFunctor : public InplaceableBinaryFunctor {
 public:
  SubFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_sub").Input("x").Input("y").Output("z").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& other, const Scalar& alpha,
                           bool inplace) const {
    if (IsIntegralDataType(input->dtype()->data_type())
        && IsIntegralDataType(other->dtype()->data_type()) && alpha.IsFloatingPoint()) {
      return Error::RuntimeError()
             << "For integral input tensors, argument alpha must not be a floating point number.";
    }
    if ((alpha.IsIntegral() && alpha.Value<int64_t>() == 1)
        || (alpha.IsFloatingPoint()
            && std::fabs(alpha.Value<double>() - 1.0) < std::numeric_limits<double>::epsilon())) {
      return InplaceableBinaryFunctor::operator()(input, other, inplace);
    } else {
      return InplaceableBinaryFunctor::operator()(input, JUST(functional::ScalarMul(alpha, other)),
                                                  inplace);
    }
  }
};

class MulFunctor {
 public:
  MulFunctor() {
    broadcast_mul_op_ =
        CHECK_JUST(one::OpBuilder("broadcast_mul").Input("x").Input("y").Output("z").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    auto tensor_x = x;
    auto tensor_y = y;
    JUST(CastDeviceForCPUScalarTensor(tensor_x, tensor_y, /*inplace=*/false));
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({tensor_x, tensor_y}).Apply());
    TensorTuple input_vec = JUST(tensor_processor.GetInputs());

    return OpInterpUtil::Dispatch<Tensor>(*broadcast_mul_op_, input_vec);
  }

 private:
  std::shared_ptr<OpExpr> broadcast_mul_op_;
};

class InplaceMulFunctor {
 public:
  InplaceMulFunctor() {
    broadcast_mul_op_ =
        CHECK_JUST(one::OpBuilder("broadcast_mul").Input("x").Input("y").Output("z").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    TensorProcessor tensor_processor;
    if (y->requires_grad()) {
      JUST(tensor_processor.PromoteInputsToCommonDtype(true)
               .AddInputs({JUST(Identity(x)), y})
               .Apply());
    } else {
      JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({x, y}).Apply());
    }
    const TensorTuple& input_vec = JUST(tensor_processor.GetInputs());
    const std::shared_ptr<one::Tensor>& x_cast = input_vec.at(0);
    const std::shared_ptr<one::Tensor>& y_cast = input_vec.at(1);
    JUST(CheckInplaceValid(x));
    JUST(CheckInplaceCastValid(x, x_cast));
    JUST(CheckInplaceShapeCanExpandTo(*y_cast->shape(), *x_cast->shape()));
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = x;
    JUST(OpInterpUtil::Dispatch(*broadcast_mul_op_, input_vec, outputs.get()));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> broadcast_mul_op_;
};

class AddcmulBaseFunctor {
 public:
  AddcmulBaseFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& tensor1,
                           const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value,
                           bool inplace) const {
    return SequenceFunction<Maybe<Tensor>()>([&]() { return functional::Mul(tensor1, tensor2); })
        .then([&](const auto& x) { return functional::ScalarMul(value, x); })
        .then([&](const auto& x) { return functional::Add(input, x, /*alpha=*/1, inplace); })
        .call();
  }
};

class AddcmulFunctor : public AddcmulBaseFunctor {
 public:
  AddcmulFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& tensor1,
                           const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value) const {
    return AddcmulBaseFunctor::operator()(input, tensor1, tensor2, value, /*inplace=*/false);
  }
};

class InplaceAddcmulFunctor : public AddcmulBaseFunctor {
 public:
  InplaceAddcmulFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& tensor1,
                           const std::shared_ptr<one::Tensor>& tensor2, const Scalar& value) const {
    return AddcmulBaseFunctor::operator()(input, tensor1, tensor2, value, /*inplace=*/true);
  }
};

class DivFunctor : public BinaryFloatFunctor {
 public:
  DivFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_div").Input("x").Input("y").Output("z").Build());
  }
};

class DivFunctorMode {
 public:
  DivFunctorMode() {}

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const Optional<std::string>& rounding_mode) const {
    std::string rmode = rounding_mode.value_or("");
    if (rmode == "floor") {
      return JUST(functional::FloorDiv(x, y));

    } else if (rmode == "trunc") {
      return JUST(functional::TruncDiv(x, y));
    }
    CHECK_OR_RETURN(rmode == "") << "div expected rounding_mode to be one of None,"
                                    " 'trunc', or 'floor' but found "
                                 << rmode;
    return JUST(functional::Div(x, y));
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class InplaceDivFunctor {
 public:
  InplaceDivFunctor() {
    broadcast_div_op_ =
        CHECK_JUST(one::OpBuilder("broadcast_div").Input("x").Input("y").Output("z").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    auto tensor_x = x;
    auto tensor_y = y;
    JUST(CastDeviceForCPUScalarTensor(tensor_x, tensor_y, /*inplace=*/true));

    // NOTE: div operator will cast inputs to float when dtype is integral
    TensorProcessor tensor_processor;
    TensorTuple tensor_processor_inputs;
    {
      if (tensor_y->requires_grad()) {
        tensor_processor_inputs.assign({JUST(Identity(tensor_x)), tensor_y});
      } else {
        tensor_processor_inputs.assign({tensor_x, tensor_y});
      }
    }
    if (promoteTypes(tensor_x->dtype(), tensor_y->dtype())->is_integer()) {
      tensor_processor.AddInputs(tensor_processor_inputs, DType::Float());
    } else {
      tensor_processor.AddInputs(tensor_processor_inputs)
          .PromoteInputsToCommonDtype(true)
          .PromoteIntegerInputsToFloatDtype(true);
    }
    JUST(tensor_processor.Apply());

    const TensorTuple& input_vec = JUST(tensor_processor.GetInputs());
    const std::shared_ptr<one::Tensor>& x_cast = input_vec.at(0);
    const std::shared_ptr<one::Tensor>& y_cast = input_vec.at(1);
    JUST(CheckInplaceValid(x));
    JUST(CheckInplaceCastValid(x, x_cast));
    JUST(CheckInplaceShapeCanExpandTo(*y_cast->shape(), *x_cast->shape()));
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = x;
    JUST(OpInterpUtil::Dispatch(*broadcast_div_op_, input_vec, outputs.get()));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> broadcast_div_op_;
};

class Atan2Functor : public BinaryFloatFunctor {
 public:
  Atan2Functor() {
    op_ = CHECK_JUST(one::OpBuilder("atan2").Input("x").Input("y").Output("z").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    const int64_t x_element = x->nelement();
    const int64_t y_element = y->nelement();
    CHECK_GT_OR_RETURN(x_element, 0)
        << Error::RuntimeError() << "the size of input should be > 0, but got " << x_element;
    CHECK_GT_OR_RETURN(y_element, 0)
        << Error::RuntimeError() << "the size of input should be > 0, but got " << y_element;

    if ((x_element != 1 && y_element != 1) && (x->shape()->NumAxes() == y->shape()->NumAxes())) {
      return BinaryFloatFunctor::operator()(x, y);
    }

    auto broad_x_ = x;
    auto broad_y_ = y;
    if (x_element == 1) {
      broad_x_ = JUST(functional::Expand(x, *y->shape()));
    } else if (y_element == 1) {
      broad_y_ = JUST(functional::Expand(y, *x->shape()));
    } else if (x->shape()->NumAxes() != y->shape()->NumAxes()) {
      return Error::RuntimeError() << "The size of tensor a (" << x->shape()->NumAxes()
                                   << ") must match the size of tensor b "
                                      "("
                                   << y->shape()->NumAxes() << ") at non-singleton dimension 1";
    } else {
      return Error::RuntimeError() << "";
    }

    return BinaryFloatFunctor::operator()(broad_x_, broad_y_);
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

class TruncDivFunctor : public BinaryFunctor {
 public:
  TruncDivFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("truncdiv").Input("x").Input("y").Output("z").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    return BinaryFunctor::operator()(x, y);
  }
};

class LerpFunctor {
 public:
  LerpFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("lerp").Input("start").Input("end").Input("weight").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& start,
                           const std::shared_ptr<one::Tensor>& end,
                           const std::shared_ptr<one::Tensor>& weight) const {
    const int64_t weight_elem_cnt = weight->nelement();
    CHECK_EQ_OR_RETURN(start->shape()->NumAxes(), end->shape()->NumAxes())
        << Error::RuntimeError() << "expected dim" << start->shape()->NumAxes()
        << "for `end` but got dim" << end->shape()->NumAxes();
    CHECK_EQ_OR_RETURN(start->dtype()->data_type(), weight->dtype()->data_type())
        << Error::RuntimeError() << "expected dtype " << start->dtype()->name()
        << " for `weights` but got dtype " << weight->dtype()->name();

    auto broadcast_shape = *start->shape();
    if (*start->shape() != *end->shape() || *start->shape() != *weight->shape()) {
      broadcast_shape = *JUST(
          InferUnifiedShapeForBroadcasting({*start->shape(), *end->shape(), *weight->shape()}));
    }

    if (weight_elem_cnt == 1 && weight->is_eager() && !weight->requires_grad()) {
      std::shared_ptr<Tensor> cast_double_weight =
          JUST(functional::Cast(weight, DType::Double(), /*pin_memory=*/false));
      double weight_scalar = JUST(GetItemInScalarTensor<double>(cast_double_weight));
      return functional::ScalarLerp(start, end, weight_scalar);
    }

    std::shared_ptr<Tensor> broadcast_start = start;
    std::shared_ptr<Tensor> broadcast_end = end;
    std::shared_ptr<Tensor> broadcast_weight = weight;
    if (*start->shape() != broadcast_shape) {
      broadcast_start = JUST(functional::Expand(start, broadcast_shape));
    }
    if (*end->shape() != broadcast_shape) {
      broadcast_end = JUST(functional::Expand(end, broadcast_shape));
    }
    if (*weight->shape() != broadcast_shape) {
      broadcast_weight = JUST(functional::Expand(weight, broadcast_shape));
    }

    return OpInterpUtil::Dispatch<Tensor>(*op_, {broadcast_start, broadcast_end, broadcast_weight});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class InplaceLerpFunctor {
 public:
  InplaceLerpFunctor() {
    lerp_op_ = CHECK_JUST(
        one::OpBuilder("lerp").Input("start").Input("end").Input("weight").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& start,
                           const std::shared_ptr<one::Tensor>& end,
                           const std::shared_ptr<one::Tensor>& weight) const {
    const int64_t weight_elem_cnt = weight->nelement();
    CHECK_EQ_OR_RETURN(start->shape()->NumAxes(), end->shape()->NumAxes())
        << Error::RuntimeError() << "expected dim" << start->shape()->NumAxes()
        << "for `end` but got dim" << end->shape()->NumAxes();
    CHECK_EQ_OR_RETURN(start->dtype()->data_type(), weight->dtype()->data_type())
        << Error::RuntimeError() << "expected dtype " << start->dtype()->name()
        << " for `weights` but got dtype " << weight->dtype()->name();

    if (weight_elem_cnt == 1 && weight->is_eager() && !weight->requires_grad()) {
      std::shared_ptr<Tensor> cast_double_weight =
          JUST(functional::Cast(weight, DType::Double(), /*pin_memory=*/false));
      double weight_scalar = JUST(GetItemInScalarTensor<double>(cast_double_weight));
      JUST(functional::ScalarInplaceLerp(start, end, weight_scalar));
      return start;
    }

    auto broadcast_shape = *start->shape();
    if (*start->shape() != *end->shape() || *start->shape() != *weight->shape()) {
      broadcast_shape = *JUST(
          InferUnifiedShapeForBroadcasting({*start->shape(), *end->shape(), *weight->shape()}));
    }

    std::shared_ptr<one::Tensor> broadcast_start = JUST(Identity(start));
    std::shared_ptr<one::Tensor> broadcast_end = JUST(Identity(end));
    std::shared_ptr<one::Tensor> broadcast_weight = JUST(Identity(weight));
    if (*start->shape() != broadcast_shape) {
      broadcast_start = JUST(view::Expand(start, broadcast_shape));
    }
    if (*end->shape() != broadcast_shape) {
      broadcast_end = JUST(view::Expand(end, broadcast_shape));
    }
    if (*weight->shape() != broadcast_shape) {
      broadcast_weight = JUST(view::Expand(weight, broadcast_shape));
    }

    TensorProcessor tensor_processor;
    if (broadcast_end->requires_grad() || broadcast_weight->requires_grad()) {
      JUST(tensor_processor.PromoteInputsToCommonDtype(true)
               .AddInputs({JUST(Identity(broadcast_start)), broadcast_end, broadcast_weight})
               .Apply());
    } else {
      JUST(tensor_processor.PromoteInputsToCommonDtype(true)
               .AddInputs({broadcast_start, broadcast_end, broadcast_weight})
               .Apply());
    }

    const TensorTuple& input_vec = JUST(tensor_processor.GetInputs());
    const std::shared_ptr<one::Tensor>& start_cast = input_vec.at(0);
    const std::shared_ptr<one::Tensor>& end_cast = input_vec.at(1);
    JUST(CheckInplaceValid(broadcast_start));
    JUST(CheckInplaceCastValid(broadcast_start, start_cast));
    JUST(CheckInplaceShapeCanExpandTo(*start_cast->shape(), *end_cast->shape()));
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = start;
    JUST(OpInterpUtil::Dispatch(*lerp_op_, input_vec, outputs.get()));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> lerp_op_;
};

class LerpGradFunctor {
 public:
  LerpGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("lerp_grad")
                         .Input("start")
                         .Input("end")
                         .Input("weight")
                         .Input("out_diff")
                         .Output("start_diff")
                         .Output("end_diff")
                         .Output("weight_diff")
                         .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& start,
                                const std::shared_ptr<one::Tensor>& end,
                                const std::shared_ptr<one::Tensor>& weight,
                                const std::shared_ptr<one::Tensor>& out_diff) const {
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {start, end, weight, out_diff}, {});
  }

 private:
  std::shared_ptr<OpExpr> op_;
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

class EqualFunctor {
 public:
  EqualFunctor() {
    broadcast_equal_op_ =
        CHECK_JUST(one::OpBuilder("broadcast_equal").Input("x").Input("y").Output("z").Build());
  }
  Maybe<bool> operator()(const std::shared_ptr<one::Tensor>& x,
                         const std::shared_ptr<one::Tensor>& y) const {
    if (*x->shape() != *y->shape()) { return false; }
    if (x->nelement() == 0) { return true; }

    std::shared_ptr<Tensor> output = JUST(
        ReduceAllWhole(JUST(OpInterpUtil::Dispatch<Tensor>(*broadcast_equal_op_, {x, y}, {}))));
    bool status = JUST(GetItemInScalarTensor<bool>(output));
    return status;
  }

 private:
  std::shared_ptr<OpExpr> broadcast_equal_op_;
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

class InplaceBroadcastGreaterFunctor {
 public:
  InplaceBroadcastGreaterFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("broadcast_inplace_greater").Input("x").Input("y").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({x, y}).Apply());
    const TensorTuple& input_vec = JUST(tensor_processor.GetInputs());
    const std::shared_ptr<one::Tensor>& x_cast = input_vec.at(0);
    const std::shared_ptr<one::Tensor>& y_cast = input_vec.at(1);
    JUST(CheckInplaceValid(x));
    JUST(CheckInplaceCastValid(x, x_cast));
    JUST(CheckInplaceShapeCanExpandTo(*y_cast->shape(), *x_cast->shape()));
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = x;
    JUST(OpInterpUtil::Dispatch(*op_, input_vec, outputs.get()));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> op_;
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

class BroadcastBitwiseAndFunctor : public BinaryFunctor {
 public:
  BroadcastBitwiseAndFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("broadcast_bitwise_and").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastBitwiseOrFunctor : public BinaryFunctor {
 public:
  BroadcastBitwiseOrFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("broadcast_bitwise_or").Input("x").Input("y").Output("z").Build());
  }
};

class BroadcastBitwiseXorFunctor : public BinaryFunctor {
 public:
  BroadcastBitwiseXorFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("broadcast_bitwise_xor").Input("x").Input("y").Output("z").Build());
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

class BroadcastIsCloseFunctor {
 public:
  BroadcastIsCloseFunctor() {
    eq_nan_op_ = CHECK_JUST(
        one::OpBuilder("broadcast_isclose_eq_nan").Input("x").Input("y").Output("z").Build());
    neq_nan_op_ = CHECK_JUST(
        one::OpBuilder("broadcast_isclose_neq_nan").Input("x").Input("y").Output("z").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y, const float atol,
                           const float rtol, const bool equal_nan) const {
    auto& attr = THREAD_CACHED_MUTABLE_ATTR_MAP("atol", "rtol", "equal_nan");
    attr.SetAllAttrs(atol, rtol, equal_nan);
    if (equal_nan) {
      return OpInterpUtil::Dispatch<Tensor>(*eq_nan_op_, {x, y}, attr);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*neq_nan_op_, {x, y}, attr);
    }
  }

 private:
  std::shared_ptr<OpExpr> eq_nan_op_;
  std::shared_ptr<OpExpr> neq_nan_op_;
};

class ScalarAddByTensorFunctor : public InplaceableBinaryFunctor {
 public:
  ScalarAddByTensorFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("scalar_add_by_tensor").Input("x").Input("scalar").Output("y").Build());
  }
};

// this functor just for test host memory input
class HostScalarAddByTensorFunctor {
 public:
  HostScalarAddByTensorFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("host_scalar_add_by_tensor").Input("x").Input("scalar").Output("y").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& scalar) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, scalar});
  }

 private:
  std::shared_ptr<OpExpr> op_;
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

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::AddFunctor>("Add");
  m.add_functor<impl::AddcmulFunctor>("Addcmul");
  m.add_functor<impl::InplaceAddcmulFunctor>("InplaceAddcmul");
  m.add_functor<impl::Atan2Functor>("Atan2");
  m.add_functor<impl::SubFunctor>("Sub");
  m.add_functor<impl::MulFunctor>("Mul");
  m.add_functor<impl::InplaceMulFunctor>("InplaceMul");
  m.add_functor<impl::InplaceDivFunctor>("InplaceDiv");
  m.add_functor<impl::DivFunctor>("Div");
  m.add_functor<impl::DivFunctorMode>("DivMode");
  m.add_functor<impl::PowFunctor>("Pow");
  m.add_functor<impl::BroadcastPowFunctor>("BroadcastPow");
  m.add_functor<impl::BroadcastEqualFunctor>("BroadcastEqual");
  m.add_functor<impl::EqualFunctor>("Equal");
  m.add_functor<impl::BroadcastNotEqualFunctor>("BroadcastNotEqual");
  m.add_functor<impl::BroadcastGreaterFunctor>("BroadcastGreater");
  m.add_functor<impl::InplaceBroadcastGreaterFunctor>("InplaceBroadcastGreater");
  m.add_functor<impl::BroadcastGreaterEqualFunctor>("BroadcastGreaterEqual");
  m.add_functor<impl::BroadcastLogicalAndFunctor>("BroadcastLogicalAnd");
  m.add_functor<impl::BroadcastLogicalOrFunctor>("BroadcastLogicalOr");
  m.add_functor<impl::BroadcastLogicalXorFunctor>("BroadcastLogicalXor");
  m.add_functor<impl::BroadcastBitwiseAndFunctor>("BroadcastBitwiseAnd");
  m.add_functor<impl::BroadcastBitwiseOrFunctor>("BroadcastBitwiseOr");
  m.add_functor<impl::BroadcastBitwiseXorFunctor>("BroadcastBitwiseXor");
  m.add_functor<impl::BroadcastLessFunctor>("BroadcastLess");
  m.add_functor<impl::BroadcastLessEqualFunctor>("BroadcastLessEqual");
  m.add_functor<impl::ScalarAddByTensorFunctor>("ScalarAddByTensor");
  m.add_functor<impl::HostScalarAddByTensorFunctor>("HostScalarAddByTensor");
  m.add_functor<impl::ScalarSubByTensorFunctor>("ScalarSubByTensor");
  m.add_functor<impl::ScalarMulByTensorFunctor>("ScalarMulByTensor");
  m.add_functor<impl::ScalarDivByTensorFunctor>("ScalarDivByTensor");
  m.add_functor<impl::BroadcastFModFunctor>("BroadcastFMod");
  m.add_functor<impl::FloorDivFunctor>("FloorDiv");
  m.add_functor<impl::TruncDivFunctor>("TruncDiv");
  m.add_functor<impl::BroadcastIsCloseFunctor>("IsClose");
  m.add_functor<impl::LerpFunctor>("Lerp");
  m.add_functor<impl::InplaceLerpFunctor>("InplaceLerp");
  m.add_functor<impl::LerpGradFunctor>("LerpGrad");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
