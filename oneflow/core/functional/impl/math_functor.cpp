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

#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/functional/tensor_processor.h"

#include <sstream>
#include <bitset>

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class AddNFunctor {
 public:
  AddNFunctor() {
    op_.resize(kMaxInputCount /*the maximum number of inputs*/);
    for (int n = 1; n < op_.size(); ++n) {
      op_[n] = CHECK_JUST(one::OpBuilder("add_n").Input("in", n + 1).Output("out").Build());
    }
  }
  Maybe<Tensor> operator()(const TensorTuple& inputs, bool inplace) const {
    CHECK_GE_OR_RETURN(inputs.size(), 2);
    TensorTuple outputs;
    for (int i = 0; i < inputs.size(); i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < inputs.size() ? kMaxInputCount : inputs.size() - i;
      TensorTuple partial_inputs(size);
      std::copy(inputs.begin() + i, inputs.begin() + i + size, partial_inputs.begin());
      if (i == 0 && inplace) {
        JUST(CheckInplaceValid(partial_inputs.at(0)));
        std::shared_ptr<TensorTuple> outs = std::make_shared<TensorTuple>(1);
        (*outs)[0] = partial_inputs[0];
        JUST(OpInterpUtil::Dispatch(*op_.at(size - 1), partial_inputs, outs.get()));
        outputs.emplace_back((*outs)[0]);
      } else {
        outputs.emplace_back(
            JUST(OpInterpUtil::Dispatch<Tensor>(*op_.at(size - 1), partial_inputs)));
      }
    }
    if (outputs.size() == 1) { return outputs.at(0); }
    return this->operator()(outputs, inplace);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> op_;
};

class ScalarMathBaseFunctor {
 public:
  explicit ScalarMathBaseFunctor(std::string op_name) {
    op_ = CHECK_JUST(one::OpBuilder(op_name).Input("in").Output("out").Build());
  }
  virtual ~ScalarMathBaseFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar,
                           bool inplace) const {
    if (std::dynamic_pointer_cast<StaticZerosTensor>(x) && op_->op_type_name() == "scalar_mul") {
      return x;
    }
    MutableAttrMap attrs;
    TensorProcessor tensor_processor;
    Symbol<DType> lowest_dtype;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      // Only promote type to Float32 when tensor is Int type but scalar is float type.
      if (DType::priority_order[x->dtype()->data_type()]
          < DType::priority_order[DType::Float16()->data_type()]) {
        lowest_dtype = DType::Float();
      } else {
        lowest_dtype = x->dtype();
      }
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      // Only promote type to Int64 when tensor is Bool type but scalar is int type.
      if (DType::priority_order[x->dtype()->data_type()]
          == DType::priority_order[DType::Bool()->data_type()]) {
        lowest_dtype = DType::Int64();
      } else {
        lowest_dtype = x->dtype();
      }
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in " << op_->op_type_name()
                                  << " should be float or int.";
    }
    JUST(tensor_processor.AddInputs({x}, lowest_dtype).Apply());
    TensorTuple casted_vec = JUST(tensor_processor.GetInputs());
    if (inplace) {
      JUST(CheckInplaceCastValid(x, casted_vec[0]));
      JUST(CheckInplaceValid(x));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = x;
      JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get(), attrs));
      return outputs->at(0);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, casted_vec, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ScalarAddFunctor : public ScalarMathBaseFunctor {
 public:
  ScalarAddFunctor() : ScalarMathBaseFunctor(/*op_name=*/"scalar_add") {}

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const Scalar& other,
                           const Scalar& alpha, const bool& inplace) const {
    if (IsIntegralDataType(input->dtype()->data_type()) && other.IsIntegral()
        && alpha.IsFloatingPoint()) {
      return Error::RuntimeError()
             << "For integral input tensors, argument alpha must not be a floating point number.";
    }
    Scalar scalar;
    if (other.IsFloatingPoint() || alpha.IsFloatingPoint()) {
      scalar = Scalar(other.Value<double>() * alpha.Value<double>());
    } else {
      scalar = Scalar(other.Value<int64_t>() * alpha.Value<int64_t>());
    }
    return ScalarMathBaseFunctor::operator()(input, scalar, inplace);
  }
};

class ScalarAdd2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& input, const std::shared_ptr<one::Tensor>& other,
                           const Scalar& alpha) const {
    if (IsIntegralDataType(other->dtype()->data_type()) && input.IsIntegral()
        && alpha.IsFloatingPoint()) {
      return Error::RuntimeError()
             << "For integral input tensors, argument alpha must not be a floating point number.";
    }
    std::shared_ptr<one::Tensor> other_;
    if ((alpha.IsIntegral() && alpha.Value<int64_t>() == 1)
        || (alpha.IsFloatingPoint()
            && std::fabs(alpha.Value<double>() - 1.0) < std::numeric_limits<double>::epsilon())) {
      other_ = other;
    } else {
      other_ = JUST(ScalarMul(alpha, other));
    }
    return ScalarAdd(other_, input, /*alpha=*/1, /*inplace=*/false);
  }
};

class ScalarSubFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar,
                           bool inplace) const {
    return ScalarAdd(x, Scalar(-1) * scalar, /*alpha=*/1, inplace);
  }
};

class ScalarSub2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarAdd(JUST(ScalarMul(x, Scalar(-1), false)), scalar, /*alpha=*/1,
                     /*inplace=*/false);
  }
};

class ScalarMulFunctor : public ScalarMathBaseFunctor {
 public:
  ScalarMulFunctor() : ScalarMathBaseFunctor(/*op_name=*/"scalar_mul") {}
};

class ScalarMul2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarMul(x, scalar, false);
  }
};

class InplaceScalarMulFunctor : public ScalarMathBaseFunctor {
 public:
  InplaceScalarMulFunctor() : ScalarMathBaseFunctor(/*op_name=*/"scalar_mul") {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    return ScalarMathBaseFunctor::operator()(x, scalar, true);
  }
};

class ScalarDivFunctor : public ScalarMathBaseFunctor {
 public:
  ScalarDivFunctor() : ScalarMathBaseFunctor(/*op_name=*/"scalar_div") {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    return ScalarMathBaseFunctor::operator()(x, scalar, false);
  }
};

class ScalarDiv2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return functional::ScalarMul(JUST(functional::ReciprocalNoNan(x)), scalar, /*inplace=*/false);
  }
};

class InplaceScalarDivFunctor : public ScalarMathBaseFunctor {
 public:
  InplaceScalarDivFunctor() : ScalarMathBaseFunctor(/*op_name=*/"scalar_mul") {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    return ScalarMathBaseFunctor::operator()(x, Scalar(1.0) / scalar, true);
  }
};

class ScalarPowFunctor : public ScalarMathBaseFunctor {
 public:
  ScalarPowFunctor() : ScalarMathBaseFunctor(/*op_name=*/"scalar_pow") {}
};

class ScalarPowGradFunctor {
 public:
  ScalarPowGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scalar_pow_grad").Input("x").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& dy, const Scalar& scalar) const {
    MutableAttrMap attrs;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in ScalarPowGrad should be float or int.";
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ScalarReversePowFunctor : public ScalarMathBaseFunctor {
 public:
  ScalarReversePowFunctor() : ScalarMathBaseFunctor(/*op_name=*/"scalar_reverse_pow") {}
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& input) const {
    return ScalarMathBaseFunctor::operator()(input, scalar, false);
  }
};

class ScalarReversePowGradFunctor {
 public:
  ScalarReversePowGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("scalar_reverse_pow_grad").Input("x").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& dy, const Scalar& scalar) const {
    MutableAttrMap attrs;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in ScalarTensorPowGrad should be float or int.";
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ScalarFloorDivFunctor : public ScalarMathBaseFunctor {
 public:
  ScalarFloorDivFunctor() : ScalarMathBaseFunctor(/*op_name=*/"scalar_floordiv") {}
};

class ScalarFModFunctor : public ScalarMathBaseFunctor {
 public:
  ScalarFModFunctor() : ScalarMathBaseFunctor(/*op_name=*/"scalar_fmod") {}
};

class ReduceMaxFunctor {
 public:
  ReduceMaxFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("reduce_max").Input("input_tensor").Output("output_tensor").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis,
                           const bool& keepdims) const {
    MutableAttrMap attrs;
    if (axis.empty()) {
      std::vector<int32_t> reduce_axis(x->shape()->NumAxes());
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", reduce_axis));
    } else {
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    }
    JUST(attrs.SetAttr<bool>("keepdims", keepdims));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ReduceMinFunctor {
 public:
  ReduceMinFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("reduce_min").Input("input_tensor").Output("output_tensor").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis,
                           const bool& keepdims) const {
    MutableAttrMap attrs;
    if (axis.empty()) {
      std::vector<int32_t> reduce_axis(x->shape()->NumAxes());
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", reduce_axis));
    } else {
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    }
    JUST(attrs.SetAttr<bool>("keepdims", keepdims));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class MaxFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    std::vector<int32_t> axis(x->ndim());
    std::iota(axis.begin(), axis.end(), 0);
    return ReduceMax(x, axis, /*keepdims=*/false);
  }
};

class Max2Functor {
 public:
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t& dim,
                                const bool& keepdims) const {
    auto outputs = std::make_shared<TensorTuple>(2);
    int32_t axis = dim;
    if (axis < -x->ndim() || axis >= x->ndim()) {
      return Error::IndexError() << "Dimension out of range (expected to be in range of ["
                                 << -x->ndim() << ", " << x->ndim() - 1 << "], but got " << axis
                                 << ")";
    }
    if (axis < 0) { axis += x->ndim(); }
    (*outputs)[0] = JUST(ReduceMax(x, {axis}, keepdims));
    (*outputs)[1] = JUST(ArgMax(x, dim, keepdims, NullOpt));
    return outputs;
  }
};

class MinFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    std::vector<int32_t> axis(x->ndim());
    std::iota(axis.begin(), axis.end(), 0);
    return ReduceMin(x, axis, /*keepdims=*/false);
  }
};

class Min2Functor {
 public:
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t& dim,
                                const bool& keepdims) const {
    auto outputs = std::make_shared<TensorTuple>(2);
    int32_t axis = dim;
    if (axis < -x->ndim() || axis >= x->ndim()) {
      return Error::IndexError() << "Dimension out of range (expected to be in range of ["
                                 << -x->ndim() << ", " << x->ndim() - 1 << "], but got " << axis
                                 << ")";
    }
    if (axis < 0) { axis += x->ndim(); }
    (*outputs)[0] = JUST(ReduceMin(x, {axis}, keepdims));
    (*outputs)[1] = JUST(ArgMin(x, dim, keepdims, NullOpt));
    return outputs;
  }
};

class AmaxFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const Optional<std::vector<int32_t>>& dim, const bool& keepdim) const {
    if (!dim.has_value()) { return ReduceMax(x, {}, keepdim); }

    const int32_t ndim = x->ndim();
    std::vector<int32_t>& dims = *JUST(dim);
    for (int i = 0; i < dims.size(); i++) {
      if (dims[i] < -ndim || dims[i] >= ndim) {
        return Error::IndexError() << "Dimension out of range (expected to be in range of ["
                                   << -ndim << ", " << ndim - 1 << "], but got " << dims[i] << ")";
      }
      if (dims[i] < 0) { dims[i] += ndim; }
    }
    return ReduceMax(x, dims, keepdim);
  }
};

class ReduceSumFunctor {
 public:
  ReduceSumFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("reduce_sum").Input("input_tensor").Output("output_tensor").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis,
                           const bool& keepdims) const {
    // const DataType dtype = x->dtype()->data_type();
    MutableAttrMap attrs;
    if (axis.empty()) {
      std::vector<int32_t> reduce_axis(x->shape()->NumAxes());
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", reduce_axis));
    } else {
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    }
    JUST(attrs.SetAttr<bool>("keepdims", keepdims));
    TensorProcessor tensor_processor;
    JUST(tensor_processor.AddInputs({x}, /*lowest_dtype=*/DType::Int64()).Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<Tensor>(*op_, input_tuple, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ReduceAllFunctor {
 public:
  ReduceAllFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("reduce_all").Input("input_tensor").Output("output_tensor").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis,
                           const bool& keepdims) const {
    MutableAttrMap attrs;
    if (axis.empty()) {
      std::vector<int32_t> reduce_axis(x->shape()->NumAxes());
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", reduce_axis));
    } else {
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    }
    JUST(attrs.SetAttr<bool>("keepdims", keepdims));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ReduceAnyFunctor {
 public:
  ReduceAnyFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("reduce_any").Input("input_tensor").Output("output_tensor").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis,
                           const bool& keepdims) const {
    MutableAttrMap attrs;
    if (axis.empty()) {
      std::vector<int32_t> reduce_axis(x->shape()->NumAxes());
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", reduce_axis));
    } else {
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    }
    JUST(attrs.SetAttr<bool>("keepdims", keepdims));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

template<class T>
class ReduceDeviceStageBaseFunctor {
 public:
  ReduceDeviceStageBaseFunctor()
      : op_(CHECK_JUST(one::OpBuilder(T::GetOpName())
                           .Input("in")
                           .Output("out")
                           .Output("mask")
                           .Output("count")
                           .Build())) {}
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& in,
                                const std::vector<int32_t>& axis) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {in}, attrs);
  }
  virtual ~ReduceDeviceStageBaseFunctor() = default;

 private:
  std::shared_ptr<OpExpr> op_;
};

template<class T>
class ReduceDeviceStageGradBaseFunctor {
 public:
  ReduceDeviceStageGradBaseFunctor()
      : op_(CHECK_JUST(one::OpBuilder(T::GetOpName())
                           .Input("out_diff")
                           .Input("mask")
                           .Input("count")
                           .Output("in_diff")
                           .Build())) {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& out_diff,
                           const std::shared_ptr<one::Tensor>& mask,
                           const std::shared_ptr<one::Tensor>& count,
                           const std::vector<int32_t>& axis) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {out_diff, mask, count}, attrs);
  }
  virtual ~ReduceDeviceStageGradBaseFunctor() = default;

 private:
  std::shared_ptr<OpExpr> op_;
};

class ReduceMinDeviceStageFunctor
    : public ReduceDeviceStageBaseFunctor<ReduceMinDeviceStageFunctor> {
 public:
  static std::string GetOpName() { return "reduce_min_device_stage"; }
};

class ReduceMaxDeviceStageFunctor
    : public ReduceDeviceStageBaseFunctor<ReduceMaxDeviceStageFunctor> {
 public:
  static std::string GetOpName() { return "reduce_max_device_stage"; }
};

class ReduceMinDeviceStageGradFunctor
    : public ReduceDeviceStageGradBaseFunctor<ReduceMinDeviceStageGradFunctor> {
 public:
  static std::string GetOpName() { return "reduce_min_device_stage_grad"; }
};

class ReduceMaxDeviceStageGradFunctor
    : public ReduceDeviceStageGradBaseFunctor<ReduceMaxDeviceStageGradFunctor> {
 public:
  static std::string GetOpName() { return "reduce_max_device_stage_grad"; }
};

template<class T>
class ReduceGlobalStageBaseFunctor {
 public:
  ReduceGlobalStageBaseFunctor()
      : op_(CHECK_JUST(one::OpBuilder(T::GetOpName())
                           .Input("in")
                           .Input("device_count")
                           .Output("out")
                           .Output("mask")
                           .Build())) {}
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& in,
                                const std::shared_ptr<one::Tensor>& device_count,
                                const std::vector<int32_t>& axis, const bool& keepdims) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    JUST(attrs.SetAttr<bool>("keepdims", keepdims));
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {in, device_count}, attrs);
  }
  virtual ~ReduceGlobalStageBaseFunctor() = default;

 private:
  std::shared_ptr<OpExpr> op_;
};

template<class T>
class ReduceGlobalStageGradBaseFunctor {
 public:
  ReduceGlobalStageGradBaseFunctor()
      : op_(CHECK_JUST(one::OpBuilder(T::GetOpName())
                           .Input("out_diff")
                           .Input("mask")
                           .Input("device_count")
                           .Output("in_diff")
                           .Build())) {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& out_diff,
                           const std::shared_ptr<one::Tensor>& mask,
                           const std::shared_ptr<one::Tensor>& device_count,
                           const std::vector<int32_t>& axis, const bool& keepdims) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    JUST(attrs.SetAttr<bool>("keepdims", keepdims));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {out_diff, mask, device_count}, attrs);
  }
  virtual ~ReduceGlobalStageGradBaseFunctor() = default;

 private:
  std::shared_ptr<OpExpr> op_;
};

class ReduceMinGlobalStageFunctor
    : public ReduceGlobalStageBaseFunctor<ReduceMinGlobalStageFunctor> {
 public:
  static std::string GetOpName() { return "reduce_min_global_stage"; }
};

class ReduceMinGlobalStageGradFunctor
    : public ReduceGlobalStageGradBaseFunctor<ReduceMinGlobalStageGradFunctor> {
 public:
  static std::string GetOpName() { return "reduce_min_global_stage_grad"; }
};

class ReduceMaxGlobalStageFunctor
    : public ReduceGlobalStageBaseFunctor<ReduceMaxGlobalStageFunctor> {
 public:
  static std::string GetOpName() { return "reduce_max_global_stage"; }
};

class ReduceMaxGlobalStageGradFunctor
    : public ReduceGlobalStageGradBaseFunctor<ReduceMaxGlobalStageGradFunctor> {
 public:
  static std::string GetOpName() { return "reduce_max_global_stage_grad"; }
};

class ReduceMeanFunctor {
 public:
  ReduceMeanFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis,
                           const bool& keepdims) const {
    // ReduceMean only calculate floating values.
    CHECK_OR_RETURN(IsFloatingDataType(x->dtype()->data_type()))
        << "RuntimeError: Can only calculate the mean of floating types.";
    const auto& sum = JUST(functional::ReduceSum(x, axis, keepdims));
    size_t reduce_count = 1;
    if (axis.empty()) {
      reduce_count = x->shape()->Count(0);
    } else {
      for (const int32_t& i : axis) { reduce_count *= x->shape()->At(i); }
    }
    if (reduce_count == 1 || reduce_count == 0) { return sum; }
    CHECK_GT_OR_RETURN(reduce_count, 0);
    return functional::ScalarMul(sum, 1.0 / reduce_count, false);
  }
};

class ReduceProdFunctor {
 public:
  ReduceProdFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("reduce_prod").Input("input_tensor").Output("output_tensor").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int32_t>& axis,
                           const bool& keepdims, const Optional<Symbol<DType>>& dtype) const {
    MutableAttrMap attrs;
    std::shared_ptr<one::Tensor> tensor = x;
    if (dtype.has_value() && (dtype != x->dtype())) { tensor = JUST(Cast(tensor, JUST(dtype))); }
    TensorProcessor tensor_processor;
    Symbol<DType> lowest_dtype;
    if (DType::priority_order[tensor->dtype()->data_type()]
        == DType::priority_order[DType::Bool()->data_type()]) {
      lowest_dtype = DType::Int64();
    } else {
      lowest_dtype = tensor->dtype();
    }
    JUST(tensor_processor.AddInputs({tensor}, lowest_dtype).Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    if (axis.empty()) {
      std::vector<int32_t> reduce_axis(tensor->shape()->NumAxes());
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", reduce_axis));
    } else {
      JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    }
    JUST(attrs.SetAttr<bool>("keepdims", keepdims));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_, input_tuple, attrs));
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TransposeFunctor {
 public:
  TransposeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("transpose").Input("input").Output("output").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::vector<int32_t>& permute) const {
    MutableAttrMap attrs;
    auto ndim = input->ndim();
    CHECK_EQ_OR_RETURN(ndim, permute.size()) << "number of dims don't match in permute";

    // handle negative permute value here, because of permute is const,
    // so copy it to local var and do modification.
    auto positive_perm = permute;
    for (auto i = 0; i < positive_perm.size(); i++) {
      if (positive_perm[i] < 0) { positive_perm[i] += ndim; }
      CHECK_OR_RETURN(positive_perm[i] >= 0 && positive_perm[i] < ndim)
          << "IndexError: Dimension out of range (expected to be in range of [" << -ndim << ","
          << ndim << " ) but got " << positive_perm[i];
    }
    // currently, view only support eager and local mode
    if (view::IsViewApplicable(input)) { return JUST(view::Transpose(input, positive_perm)); }
    JUST(attrs.SetAttr<std::vector<int32_t>>("perm", positive_perm));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class Transpose2dimFunctor {
 public:
  Transpose2dimFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("transpose").Input("input").Output("output").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t dim0,
                           const int32_t dim1) const {
    MutableAttrMap attrs;
    const int64_t ndim = input->shape()->NumAxes();
    std::vector<int32_t> permute;
    permute.reserve(ndim);
    int32_t dim_0 = dim0;
    int32_t dim_1 = dim1;

    if (dim0 < 0) { dim_0 += ndim; }
    if (dim1 < 0) { dim_1 += ndim; }

    CHECK_OR_RETURN(dim_0 >= 0 && dim0 < ndim)
        << "Dimension out of range (expected to be in range of [" << -ndim << ", " << ndim - 1
        << "], but got " << dim_0 << ")";
    CHECK_OR_RETURN(dim_1 >= 0 && dim1 < ndim)
        << "Dimension out of range (expected to be in range of [" << -ndim << ", " << ndim - 1
        << "], but got " << dim_1 << ")";
    for (int32_t i = 0; i < ndim; ++i) { permute.emplace_back(i); }
    std::swap(permute[dim_0], permute[dim_1]);
    Shape shape(DimVector(permute.begin(), permute.end()));
    if (view::IsViewApplicable(input)) { return JUST(view::Transpose(input, permute)); }
    JUST(attrs.SetAttr<std::vector<int32_t>>("perm", permute));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class AsStridedFunctor {
 public:
  AsStridedFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("as_strided").Input("input").Output("output").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::vector<int32_t>& size, const std::vector<int32_t>& stride,
                           const int32_t& storage_offset) const {
    CHECK_OR_RETURN(size.size() == stride.size()) << "mismatch in length of strides and shape";
    for (size_t i = 0; i < size.size(); i++) {
      CHECK_OR_RETURN(size[i] >= 0) << "Trying to create tensor with negative dimension" << size[i];
      CHECK_OR_RETURN(stride[i] >= 0)
          << "as_strided: Negative strides are not supported at the moment, got strides:"
          << stride[i];
    }
    if (view::IsViewApplicable(input)) {
      return JUST(view::AsStrided(input, size, stride, storage_offset));
    }
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("size", size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("stride", stride));
    JUST(attrs.SetAttr<int32_t>("storage_offset", storage_offset));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class AsStridedGradFunctor {
 public:
  AsStridedGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("as_strided_grad").Input("dy").Input("input").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& input,
                           const std::vector<int32_t>& size, const std::vector<int32_t>& stride,
                           const int32_t& storage_offset) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("size", size));
    JUST(attrs.SetAttr<std::vector<int32_t>>("stride", stride));
    JUST(attrs.SetAttr<int32_t>("storage_offset", storage_offset));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ArangeFunctor {
 public:
  ArangeFunctor() { op_ = CHECK_JUST(one::OpBuilder("arange").Output("out").Build()); }
  Maybe<Tensor> operator()(const Scalar& start, const Scalar& limit, const Scalar& delta,
                           const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device) const {
    MutableAttrMap attrs;
    if (dtype.has_value()) {
      const DataType range_dtype = JUST(dtype)->data_type();
      if (IsIntegralDataType(range_dtype)) {
        JUST(attrs.SetAttr<int64_t>("integer_start", JUST(start.As<int64_t>())));
        JUST(attrs.SetAttr<int64_t>("integer_limit", JUST(limit.As<int64_t>())));
        JUST(attrs.SetAttr<int64_t>("integer_delta", JUST(delta.As<int64_t>())));
        JUST(attrs.SetAttr<DataType>("dtype", range_dtype));
      } else {
        JUST(attrs.SetAttr<double>("float_start", JUST(start.As<double>())));
        JUST(attrs.SetAttr<double>("float_limit", JUST(limit.As<double>())));
        JUST(attrs.SetAttr<double>("float_delta", JUST(delta.As<double>())));
        JUST(attrs.SetAttr<DataType>("dtype", range_dtype));
      }
    } else {
      if (delta.IsIntegral()) {
        JUST(attrs.SetAttr<int64_t>("integer_start", JUST(start.As<int64_t>())));
        JUST(attrs.SetAttr<int64_t>("integer_limit", JUST(limit.As<int64_t>())));
        JUST(attrs.SetAttr<int64_t>("integer_delta", JUST(delta.As<int64_t>())));
        JUST(attrs.SetAttr<DataType>("dtype", DType::Int64()->data_type()));
      } else {
        JUST(attrs.SetAttr<double>("float_start", JUST(start.As<double>())));
        JUST(attrs.SetAttr<double>("float_limit", JUST(limit.As<double>())));
        JUST(attrs.SetAttr<double>("float_delta", JUST(delta.As<double>())));
        JUST(attrs.SetAttr<DataType>("dtype", DType::Float()->data_type()));
      }
    }
    OpExprInterpContext ctx(attrs);
    ctx.device = device;
    return OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class Arange2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& limit, const Optional<Symbol<DType>>& dtype,
                           const Optional<Symbol<Device>>& device) const {
    return Arange(Scalar(0), limit, Scalar(1), dtype, device);
  }
};

class ConsistentArangeFunctor {
 public:
  ConsistentArangeFunctor() { op_ = CHECK_JUST(one::OpBuilder("arange").Output("out").Build()); }
  Maybe<Tensor> operator()(const Scalar& start, const Scalar& limit, const Scalar& delta,
                           const Optional<Symbol<DType>>& dtype,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    JUST(CheckDeviceIdsIsValid(placement));
    MutableAttrMap attrs;
    if (dtype.has_value()) {
      const DataType range_dtype = JUST(dtype)->data_type();
      if (IsIntegralDataType(range_dtype)) {
        JUST(attrs.SetAttr<int64_t>("integer_start", JUST(start.As<int64_t>())));
        JUST(attrs.SetAttr<int64_t>("integer_limit", JUST(limit.As<int64_t>())));
        JUST(attrs.SetAttr<int64_t>("integer_delta", JUST(delta.As<int64_t>())));
        JUST(attrs.SetAttr<DataType>("dtype", range_dtype));
      } else {
        JUST(attrs.SetAttr<double>("float_start", JUST(start.As<double>())));
        JUST(attrs.SetAttr<double>("float_limit", JUST(limit.As<double>())));
        JUST(attrs.SetAttr<double>("float_delta", JUST(delta.As<double>())));
        JUST(attrs.SetAttr<DataType>("dtype", range_dtype));
      }
    } else {
      if (delta.IsIntegral()) {
        JUST(attrs.SetAttr<int64_t>("integer_start", JUST(start.As<int64_t>())));
        JUST(attrs.SetAttr<int64_t>("integer_limit", JUST(limit.As<int64_t>())));
        JUST(attrs.SetAttr<int64_t>("integer_delta", JUST(delta.As<int64_t>())));
        JUST(attrs.SetAttr<DataType>("dtype", DType::Int64()->data_type()));
      } else {
        JUST(attrs.SetAttr<double>("float_start", JUST(start.As<double>())));
        JUST(attrs.SetAttr<double>("float_limit", JUST(limit.As<double>())));
        JUST(attrs.SetAttr<double>("float_delta", JUST(delta.As<double>())));
        JUST(attrs.SetAttr<DataType>("dtype", DType::Float()->data_type()));
      }
    }
    if (LazyMode::is_enabled()) {
      std::vector<std::string> nd_sbp(sbp_tuple.size());
      {
        for (int i = 0; i < sbp_tuple.size(); ++i) {
          nd_sbp.at(i) = SbpParallelToString(*sbp_tuple.at(i));
        }
      }
      JUST(attrs.SetAttr<std::vector<std::string>>("nd_sbp", nd_sbp));
    }
    const auto& nd_sbp = JUST(GetNdSbp(sbp_tuple));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {}, OpExprInterpContext(attrs, placement, nd_sbp));
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConsistentArange2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& limit, const Symbol<DType>& dtype,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<SbpParallel>>& sbp_tuple) const {
    JUST(CheckDeviceIdsIsValid(placement));
    return ConsistentArange(Scalar(0), limit, Scalar(1), dtype, placement, sbp_tuple);
  }
};

class CastFunctor {
 public:
  CastFunctor() { op_ = CHECK_JUST(one::OpBuilder("cast").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const Symbol<DType>& dtype) const {
    if (x->dtype() == dtype) { return x; }

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ClampBaseFunctor {
 public:
  ClampBaseFunctor() {
    clip_op_ = CHECK_JUST(one::OpBuilder("clip_by_scalar").Input("x").Output("y").Build());
    clip_min_op_ = CHECK_JUST(one::OpBuilder("clip_by_scalar_min").Input("x").Output("y").Build());
    clip_max_op_ = CHECK_JUST(one::OpBuilder("clip_by_scalar_max").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Optional<Scalar>& min,
                           const Optional<Scalar>& max, bool inplace) const {
    CHECK_OR_RETURN(min.has_value() || max.has_value())
        << "Requires one of argument `min` and `max` at least in clip.";
    MutableAttrMap attrs;
    if (IsFloatingDataType(x->dtype()->data_type())) {
      if (min.has_value()) {
        const auto& min_val = JUST(min);
        JUST(attrs.SetAttr<double>("floating_min", JUST(min_val->As<double>())));
        JUST(attrs.SetAttr<int64_t>("integral_min", 0));
      }
      if (max.has_value()) {
        const auto& max_val = JUST(max);
        JUST(attrs.SetAttr<double>("floating_max", JUST(max_val->As<double>())));
        JUST(attrs.SetAttr<int64_t>("integral_max", 0));
      }
    } else if (IsIntegralDataType(x->dtype()->data_type())) {
      if (min.has_value()) {
        const auto& min_val = JUST(min);
        JUST(attrs.SetAttr<double>("floating_min", 0));
        JUST(attrs.SetAttr<int64_t>("integral_min", JUST(min_val->As<int64_t>())));
      }
      if (max.has_value()) {
        const auto& max_val = JUST(max);
        JUST(attrs.SetAttr<double>("floating_max", 0));
        JUST(attrs.SetAttr<int64_t>("integral_max", JUST(max_val->As<int64_t>())));
      }
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Only support floating or integral data type.";
    }
    const OpExpr* op = nullptr;
    if (!min.has_value()) {
      op = clip_max_op_.get();
    } else if (!max.has_value()) {
      op = clip_min_op_.get();
    } else {
      op = clip_op_.get();
    }
    if (inplace) {
      JUST(CheckInplaceValid(x));
      std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
      outputs->at(0) = x;
      if (x->requires_grad()) {
        JUST(OpInterpUtil::Dispatch(*op, {JUST(functional::Identity(x))}, outputs.get(), attrs));
      } else {
        JUST(OpInterpUtil::Dispatch(*op, {x}, outputs.get(), attrs));
      }
      return outputs->at(0);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op, {x}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> clip_op_;
  std::shared_ptr<OpExpr> clip_min_op_;
  std::shared_ptr<OpExpr> clip_max_op_;
};

class ClampFunctor : public ClampBaseFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Optional<Scalar>& min,
                           const Optional<Scalar>& max) const {
    return ClampBaseFunctor::operator()(x, min, max, false);
  }
};

class ClampInplaceFunctor : public ClampBaseFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Optional<Scalar>& min,
                           const Optional<Scalar>& max) const {
    return ClampBaseFunctor::operator()(x, min, max, true);
  }
};

class ClipFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Optional<Scalar>& min,
                           const Optional<Scalar>& max) const {
    return Clamp(x, min, max);
  }
};

class ClipInplaceFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Optional<Scalar>& min,
                           const Optional<Scalar>& max) const {
    return ClampInplace(x, min, max);
  }
};
class SqrtSquareSumFunctor {
 public:
  SqrtSquareSumFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("sqrt_square_sum").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, {});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class VectorNormFunctor {
 public:
  VectorNormFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& ord,
                           const Optional<std::vector<int32_t>>& input_dim, const bool& keepdim,
                           const Optional<Symbol<DType>>& dtype) const {
    std::shared_ptr<one::Tensor> res;
    Symbol<DType> dtype_val;
    if (dtype) {
      dtype_val = JUST(dtype);
      if (!(dtype_val->data_type() == DataType::kFloat
            || dtype_val->data_type() == DataType::kDouble
            || dtype_val->data_type() == DataType::kFloat16
            || dtype_val->data_type() == DataType::kBFloat16)) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.vector_norm(): only supports floating point and "
                                       "complex dtypes, but got: Int.";
      }
    } else {
      if (!IsFloatingDataType(x->dtype()->data_type())) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.vector_norm(): only supports floating point and "
                                       "complex dtypes, but got: Int.";
      }
      dtype_val = x->dtype();
    }
    bool full_dim_flag = true;
    std::vector<int32_t> dim;
    if (!input_dim.has_value()) {
      std::vector<int32_t> reduce_axis(x->shape()->NumAxes());
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
      dim = reduce_axis;
    } else {
      std::vector<int32_t> dim_check;
      dim_check = *JUST(input_dim);
      for (int i = 0; i < dim_check.size(); ++i) {
        if (dim_check[i] >= 0) {
          dim.emplace_back(dim_check[i]);
        } else {
          dim.emplace_back(dim_check[i] + x->shape()->NumAxes());
        }
        if (dim[i] != i) { full_dim_flag = false; }
      }
      if ((int)dim.size() < x->shape()->NumAxes()) { full_dim_flag = false; }
    }
    if (ord.IsIntegral() || ord.IsFloatingPoint()) {
      double ord_val = JUST(ord.As<double>());
      if (ord_val == 0) {
        res = JUST(ReduceSum(JUST(functional::NotEqualZero(x)), dim, keepdim));
      } else if (ord_val == INFINITY) {
        res = JUST(ReduceMax(JUST(Abs(x)), dim, keepdim));
      } else if (ord_val == -INFINITY) {
        res = JUST(ReduceMin(JUST(Abs(x)), dim, keepdim));
      } else if (ord_val == 2.0 && keepdim == false && full_dim_flag
                 && x->requires_grad() == false) {
        res = JUST(SqrtSquareSum(x));
      } else {
        res =
            JUST(ScalarPow(JUST(ReduceSum(JUST(ScalarPow(JUST(Abs(x)), ord, false)), dim, keepdim)),
                           Scalar(1.0) / ord, false));
      }
      res = JUST(Cast(res, dtype_val));
      return res;
    } else {
      UNIMPLEMENTED_THEN_RETURN()
          << "linalg_vector_norm(): argument 'ord' must be Number, not str.";
    }
  }
};

class ScalarVectorNormFunctor {
 public:
  ScalarVectorNormFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& ord,
                           const Scalar& input_dim, const bool& keepdim,
                           const Optional<Symbol<DType>>& dtype) const {
    if (dtype) {
      Symbol<DType> dtype_val = JUST(dtype);
      if (!(dtype_val->data_type() == DataType::kFloat
            || dtype_val->data_type() == DataType::kDouble
            || dtype_val->data_type() == DataType::kFloat16
            || dtype_val->data_type() == DataType::kBFloat16)) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.vector_norm(): only supports the float, double, "
                                       "cfloat and cdouble dtypes, but got: Int.";
      }
    } else {
      if (!IsFloatingDataType(x->dtype()->data_type())) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.vector_norm(): only supports the float, double, "
                                       "cfloat and cdouble dtypes, but got: Int.";
      }
    }
    if (input_dim.IsIntegral()) {
      std::vector<int32_t> dim(1, JUST(input_dim.As<int>()));
      return functional::VectorNorm(x, ord, dim, keepdim, dtype);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "linalg.vector_norm(): only support int dim.";
    }
  }
};

class ScalarMatrixNormFunctor {
 public:
  ScalarMatrixNormFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& ord,
                           const std::vector<int32_t>& input_dim, const bool& keepdim,
                           const Optional<Symbol<DType>>& dtype) const {
    std::shared_ptr<one::Tensor> res;

    auto num_dims = x->shape()->NumAxes();
    auto axis = input_dim.size();
    CHECK_OR_RETURN(num_dims >= 2)
        << "linalg.matrix_norm(): input tensor must be a matrix or batch of matrices";
    CHECK_OR_RETURN(axis == 2 && input_dim[0] != input_dim[1])
        << "linalg.matrix_norm(): input_dim must be a 2-tuple of ints with different elements";

    Symbol<DType> dtype_val;
    if (dtype) {
      dtype_val = JUST(dtype);
      if (!(dtype_val->data_type() == DataType::kFloat
            || dtype_val->data_type() == DataType::kDouble
            || dtype_val->data_type() == DataType::kFloat16
            || dtype_val->data_type() == DataType::kBFloat16)) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.matrix_norm(): only supports the float, double, "
                                       "cfloat and cdouble dtypes, but got: Int.";
      }
    } else {
      if (!IsFloatingDataType(x->dtype()->data_type())) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.matrix_norm(): only supports the float, double, "
                                       "cfloat and cdouble dtypes, but got: Int.";
      }
      dtype_val = x->dtype();
    }
    std::vector<int32_t> dim_tmp;
    dim_tmp.reserve(axis);
    for (int i = 0; i < axis; ++i) {
      if (input_dim[i] >= 0) {
        dim_tmp.emplace_back(input_dim[i]);
      } else {
        dim_tmp.emplace_back(input_dim[i] + num_dims);
      }
    }
    std::vector<int32_t> dim(2);
    double ord_tmp = JUST(ord.As<double>());
    if (ord_tmp == INFINITY || ord_tmp == -INFINITY) {
      dim = dim_tmp;
      dim[0] = dim_tmp[1];
      dim[1] = dim_tmp[0];
    } else if (ord_tmp == 1 || ord_tmp == -1) {
      dim = dim_tmp;
    } else {
      UNIMPLEMENTED_THEN_RETURN()
          << "linalg.matrix_norm(): Only support INFINITY,-INFINITY,1 or -1 data type.";
    }

    if (dim[1] > dim[0] && keepdim == false) { dim[1] -= 1; }
    std::vector<int32_t> dim_tmp0_vec(1, dim[0]);
    std::vector<int32_t> dim_tmp1_vec(1, dim[1]);
    res = JUST(ReduceSum(JUST(Abs(x)), dim_tmp0_vec, keepdim));

    if (ord_tmp == INFINITY || ord_tmp == 1) {
      res = JUST(ReduceMax(res, dim_tmp1_vec, keepdim));
    } else if (ord_tmp == -INFINITY || ord_tmp == -1) {
      res = JUST(ReduceMin(res, dim_tmp1_vec, keepdim));
    }
    res = JUST(Cast(res, dtype_val));
    return res;
  }
};

class MatrixNormFunctor {
 public:
  MatrixNormFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::string& ord,
                           const std::vector<int32_t>& input_dim, const bool& keepdim,
                           const Optional<Symbol<DType>>& dtype) const {
    std::shared_ptr<one::Tensor> res;
    Symbol<DType> dtype_val;
    if (dtype) {
      dtype_val = JUST(dtype);
      if (!(dtype_val->data_type() == DataType::kFloat
            || dtype_val->data_type() == DataType::kDouble
            || dtype_val->data_type() == DataType::kFloat16
            || dtype_val->data_type() == DataType::kBFloat16)) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.matrix_norm(): only supports the float, double, "
                                       "cfloat and cdouble dtypes, but got: Int.";
      }
    } else {
      if (!IsFloatingDataType(x->dtype()->data_type())) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.matrix_norm(): only supports the float, double, "
                                       "cfloat and cdouble dtypes, but got: Int.";
      }
      dtype_val = x->dtype();
    }
    auto num_dims = x->shape()->NumAxes();
    auto axis = input_dim.size();
    std::vector<int32_t> dim_tmp(axis);
    for (int i = 0; i < axis; ++i) {
      if (input_dim[i] >= 0) {
        dim_tmp[i] = input_dim[i];
      } else {
        dim_tmp[i] = input_dim[i] + num_dims;
      }
    }
    if (ord == "nuc") {
      UNIMPLEMENTED_THEN_RETURN() << "linalg.matrix_norm(): Not support ord is nuc.";
    } else if (ord == "fro") {
      res = JUST(Sqrt(JUST(ReduceSum(JUST(Square(x)), dim_tmp, keepdim))));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "linalg.matrix_norm(): could not convert string to float:"
                                  << ord;
    }
    res = JUST(Cast(res, dtype_val));
    return res;
  }
};

class NormFunctor {
 public:
  NormFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Optional<Scalar>& ord,
                           const Optional<std::vector<int32_t>>& input_dim, const bool& keepdim,
                           const Optional<Symbol<DType>>& dtype) const {
    std::shared_ptr<one::Tensor> res;
    if (dtype) {
      Symbol<DType> dtype_val = JUST(dtype);
      if (!(dtype_val->data_type() == DataType::kFloat
            || dtype_val->data_type() == DataType::kDouble
            || dtype_val->data_type() == DataType::kFloat16
            || dtype_val->data_type() == DataType::kBFloat16)) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.norm(): only supports the float, double, cfloat and "
                                       "cdouble dtypes, but got: Int.";
      }
    } else {
      if (!IsFloatingDataType(x->dtype()->data_type())) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.norm(): only supports the float, double, cfloat and "
                                       "cdouble dtypes, but got: Int.";
      }
    }
    Scalar ord_sca;
    if (ord.has_value()) {
      auto ord_type = (*JUST(ord)).IsIntegral();
      if (ord_type) {
        ord_sca = Scalar(JUST((*JUST(ord)).As<double>()));
      } else {
        ord_sca = *JUST(ord);
      }
    }
    if (input_dim.has_value()) {
      auto axis = (*JUST(input_dim)).size();
      if (axis == 1) {
        Scalar ord_val;
        if (!ord.has_value()) {
          ord_val = Scalar(2.0);
        } else {
          ord_val = ord_sca;
        }
        res = JUST(VectorNorm(x, ord_val, input_dim, keepdim, dtype));
      } else if (axis > 2) {
        res = JUST(MatrixNorm(x, ord_sca, *JUST(input_dim), keepdim, dtype));
      } else if (axis == 2) {
        if (!ord.has_value()) {
          res = JUST(MatrixNorm(x, "fro", *JUST(input_dim), keepdim, dtype));
        } else {
          res = JUST(MatrixNorm(x, ord_sca, *JUST(input_dim), keepdim, dtype));
        }
      }
    } else {
      if (ord.has_value()) {
        CHECK_OR_RETURN(x->shape()->NumAxes() <= 2)
            << "linalg.norm(): input must be 1-D or 2-D when dim is None and ord is not None";
        if (x->shape()->NumAxes() == 1) {
          res = JUST(VectorNorm(x, ord_sca, input_dim, keepdim, dtype));
        } else {
          std::vector<int32_t> dim{0, 1};
          res = JUST(MatrixNorm(x, ord_sca, dim, keepdim, dtype));
        }
      } else {
        std::vector<int32_t> dim(1, 2);
        res = JUST(VectorNorm(JUST(Flatten(x, 0, -1)), Scalar(2.0), input_dim, keepdim, dtype));
      }
    }
    return res;
  }
};

class Norm2Functor {
 public:
  Norm2Functor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::string& ord,
                           const Optional<std::vector<int32_t>>& input_dim, const bool& keepdim,
                           const Optional<Symbol<DType>>& dtype) const {
    std::shared_ptr<one::Tensor> res;
    std::vector<int32_t> dim(x->shape()->NumAxes());
    std::iota(dim.begin(), dim.end(), 0);
    if (dtype) {
      Symbol<DType> dtype_val = JUST(dtype);
      if (!(dtype_val->data_type() == DataType::kFloat
            || dtype_val->data_type() == DataType::kDouble
            || dtype_val->data_type() == DataType::kFloat16
            || dtype_val->data_type() == DataType::kBFloat16)) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.norm(): only supports the float, double, cfloat and "
                                       "cdouble dtypes, but got: Int.";
      }
    } else {
      if (!IsFloatingDataType(x->dtype()->data_type())) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.norm(): only supports the float, double, cfloat and "
                                       "cdouble dtypes, but got: Int.";
      }
    }
    if (input_dim.has_value()) {
      res = JUST(MatrixNorm(x, ord, *JUST(input_dim), keepdim, dtype));
    } else {
      res = JUST(MatrixNorm(x, ord, dim, keepdim, dtype));
    }
    return res;
  }
};

class ScalarNormFunctor {
 public:
  ScalarNormFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Optional<Scalar>& ord,
                           const Scalar& input_dim, const bool& keepdim,
                           const Optional<Symbol<DType>>& dtype) const {
    if (dtype) {
      Symbol<DType> dtype_val = JUST(dtype);
      if (!(dtype_val->data_type() == DataType::kFloat
            || dtype_val->data_type() == DataType::kDouble
            || dtype_val->data_type() == DataType::kFloat16
            || dtype_val->data_type() == DataType::kBFloat16)) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.norm(): only supports the float, double, cfloat and "
                                       "cdouble dtypes, but got: Int.";
      }
    } else {
      if (!IsFloatingDataType(x->dtype()->data_type())) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.norm(): only supports the float, double, cfloat and "
                                       "cdouble dtypes, but got: Int.";
      }
    }
    if (input_dim.IsIntegral()) {
      std::vector<int32_t> dim(1, JUST(input_dim.As<int>()));
      return functional::Norm(x, ord, dim, keepdim, dtype);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "linalg_norm(): only supports int dim.";
    }
  }
};

class ScalarNorm2Functor {
 public:
  ScalarNorm2Functor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::string& ord,
                           const Scalar& input_dim, const bool& keepdim,
                           const Optional<Symbol<DType>>& dtype) const {
    if (dtype) {
      Symbol<DType> dtype_val = JUST(dtype);
      if (!(dtype_val->data_type() == DataType::kFloat
            || dtype_val->data_type() == DataType::kDouble
            || dtype_val->data_type() == DataType::kFloat16
            || dtype_val->data_type() == DataType::kBFloat16)) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.norm(): only supports the float, double, cfloat and "
                                       "cdouble dtypes, but got: Int.";
      }
    } else {
      if (!IsFloatingDataType(x->dtype()->data_type())) {
        UNIMPLEMENTED_THEN_RETURN() << "linalg.norm(): only supports the float, double, cfloat and "
                                       "cdouble dtypes, but got: Int.";
      }
    }
    if (input_dim.IsIntegral()) {
      std::vector<int32_t> dim(1, JUST(input_dim.As<int>()));
      return functional::Norm(x, ord, dim, keepdim, dtype);
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "linalg_norm(): only supports int dim.";
    }
  }
};

class ClampGradFunctor {
 public:
  ClampGradFunctor() {
    clip_op_ = CHECK_JUST(
        one::OpBuilder("clip_by_scalar_grad").Input("dy").Input("x").Output("dx").Build());
    clip_min_op_ = CHECK_JUST(
        one::OpBuilder("clip_by_scalar_min_grad").Input("dy").Input("x").Output("dx").Build());
    clip_max_op_ = CHECK_JUST(
        one::OpBuilder("clip_by_scalar_max_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const Optional<Scalar>& min,
                           const Optional<Scalar>& max) const {
    CHECK_OR_RETURN(min.has_value() || max.has_value())
        << "Requires one of argument `min` and `max` at least in clip_grad.";
    MutableAttrMap attrs;
    if (IsFloatingDataType(x->dtype()->data_type())) {
      if (min.has_value()) {
        const auto& min_val = JUST(min);
        JUST(attrs.SetAttr<double>("floating_min", JUST(min_val->As<double>())));
        JUST(attrs.SetAttr<int64_t>("integral_min", 0));
      }
      if (max.has_value()) {
        const auto& max_val = JUST(max);
        JUST(attrs.SetAttr<double>("floating_max", JUST(max_val->As<double>())));
        JUST(attrs.SetAttr<int64_t>("integral_max", 0));
      }
    } else if (IsIntegralDataType(x->dtype()->data_type())) {
      if (min.has_value()) {
        const auto& min_val = JUST(min);
        JUST(attrs.SetAttr<int64_t>("integral_min", JUST(min_val->As<int64_t>())));
        JUST(attrs.SetAttr<double>("floating_min", 0));
      }
      if (max.has_value()) {
        const auto& max_val = JUST(max);
        JUST(attrs.SetAttr<double>("floating_max", 0));
        JUST(attrs.SetAttr<int64_t>("integral_max", JUST(max_val->As<int64_t>())));
      }
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Only support floating or integral data type.";
    }
    const OpExpr* op = nullptr;
    if (!min.has_value()) {
      op = clip_max_op_.get();
    } else if (!max.has_value()) {
      op = clip_min_op_.get();
    } else {
      op = clip_op_.get();
    }
    return OpInterpUtil::Dispatch<Tensor>(*op, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> clip_op_;
  std::shared_ptr<OpExpr> clip_min_op_;
  std::shared_ptr<OpExpr> clip_max_op_;
};

class SelectFunctor {
 public:
  SelectFunctor() = default;

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const int32_t& index) const {
    int32_t ndim = input->ndim();
    CHECK_OR_RETURN(ndim > 0) << "select() cannot be applied to a 0-dim tensor.";
    CHECK_OR_RETURN((dim >= -ndim) && (dim < ndim))
        << "Dimension out of range (expected to be in range of [" << -ndim << "," << ndim - 1
        << "], but got " << dim << ")";
    int32_t pos_dim = dim >= 0 ? dim : dim + ndim;
    auto size = input->dim(pos_dim);
    CHECK_OR_RETURN((index >= -size) && (index < size))
        << "Index out of range (expected to be in range of [" << -size << "," << size - 1
        << "], but got " << index << ")";
    int32_t pos_index = index >= 0 ? index : index + size;

    std::vector<int32_t> sizes(input->shape()->dim_vec().begin(), input->shape()->dim_vec().end());
    const auto& stride = JUST(input->stride())->StrideVec();
    std::vector<int32_t> strides(stride.begin(), stride.end());
    auto storage_offset = JUST(input->storage_offset()) + pos_index * strides[pos_dim];

    sizes.erase(sizes.begin() + pos_dim);
    strides.erase(strides.begin() + pos_dim);

    return AsStrided(input, sizes, strides, storage_offset);
  }
};

class SelectTopNFunctor {
 public:
  SelectTopNFunctor() { op_ = CHECK_JUST(one::SelectTopNOpExpr::New()); }

  Maybe<TensorTuple> operator()(const TensorTuple& inputs, int32_t n) const {
    MutableAttrMap attr;
    JUST(attr.SetAttr<int32_t>("top_n", n));
    std::vector<bool> require_grad(n);
    for (int i = 0; i < n; ++i) { require_grad[i] = JUST(VectorAt(inputs, i))->requires_grad(); }
    const auto& output = JUST(OpInterpUtil::Dispatch<one::TensorTuple>(*op_, inputs, attr));
    for (int i = 0; i < output->size(); ++i) {
      (*output)[i]->set_is_leaf(false);
      JUST((*output)[i]->set_requires_grad(require_grad[i]));
    }
    return output;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class MinimumFunctor {
 public:
  MinimumFunctor() {
    elementwise_minimum_op_ =
        CHECK_JUST(one::OpBuilder("elementwise_minimum").Input("x").Input("y").Output("z").Build());
    broadcast_minimum_op_ =
        CHECK_JUST(one::OpBuilder("broadcast_minimum").Input("x").Input("y").Output("z").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({x, y}).Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    if (*x->shape() == *y->shape()) {
      return OpInterpUtil::Dispatch<Tensor>(*elementwise_minimum_op_,
                                            {input_tuple[0], input_tuple[1]});
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*broadcast_minimum_op_,
                                            {input_tuple[0], input_tuple[1]});
    }
  }

 private:
  std::shared_ptr<OpExpr> elementwise_minimum_op_;
  std::shared_ptr<OpExpr> broadcast_minimum_op_;
};

class MaximumFunctor {
 public:
  MaximumFunctor() {
    elementwise_maximum_op_ =
        CHECK_JUST(one::OpBuilder("elementwise_maximum").Input("x").Input("y").Output("z").Build());
    broadcast_maximum_op_ =
        CHECK_JUST(one::OpBuilder("broadcast_maximum").Input("x").Input("y").Output("z").Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    TensorProcessor tensor_processor;
    JUST(tensor_processor.PromoteInputsToCommonDtype(true).AddInputs({x, y}).Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    if (*x->shape() == *y->shape()) {
      return OpInterpUtil::Dispatch<Tensor>(*elementwise_maximum_op_,
                                            {input_tuple[0], input_tuple[1]});
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*broadcast_maximum_op_,
                                            {input_tuple[0], input_tuple[1]});
    }
  }

 private:
  std::shared_ptr<OpExpr> elementwise_maximum_op_;
  std::shared_ptr<OpExpr> broadcast_maximum_op_;
};

class ScalarLogicalBaseFunctor {
 public:
  explicit ScalarLogicalBaseFunctor(std::string op_name) {
    op_ = CHECK_JUST(one::OpBuilder(op_name).Input("in").Output("out").Build());
  }
  virtual ~ScalarLogicalBaseFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    MutableAttrMap attrs;
    TensorProcessor tensor_processor;
    Symbol<DType> lowest_dtype;

    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      // Only promote type to Float32 when tensor is Int type but scalar is float type.
      if (DType::priority_order[x->dtype()->data_type()]
          < DType::priority_order[DType::Float16()->data_type()]) {
        lowest_dtype = DType::Float();
      } else {
        lowest_dtype = x->dtype();
      }
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      // Only promote type to Int64 when tensor is Bool type but scalar is int type.
      if (DType::priority_order[x->dtype()->data_type()]
          == DType::priority_order[DType::Bool()->data_type()]) {
        lowest_dtype = DType::Int64();
      } else {
        lowest_dtype = x->dtype();
      }
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in " << op_->op_type_name()
                                  << " should be float or int.";
    }
    JUST(tensor_processor.AddInputs({x}, lowest_dtype).Apply());
    TensorTuple casted_vec = JUST(tensor_processor.GetInputs());

    return OpInterpUtil::Dispatch<Tensor>(*op_, {casted_vec}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ScalarLogicalEqualFunctor : public ScalarLogicalBaseFunctor {
 public:
  ScalarLogicalEqualFunctor() : ScalarLogicalBaseFunctor(/*op_name=*/"scalar_logical_equal") {}
};

// (scalar == x) = (x == scalar)
class ScalarLogicalEqual2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarLogicalEqual(x, scalar);
  }
};

class ScalarLogicalNotEqualFunctor : public ScalarLogicalBaseFunctor {
 public:
  ScalarLogicalNotEqualFunctor()
      : ScalarLogicalBaseFunctor(/*op_name=*/"scalar_logical_not_equal") {}
};

// (scalar != x) = (x != scalar)
class ScalarLogicalNotEqual2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarLogicalNotEqual(x, scalar);
  }
};

class ScalarLogicalGreaterFunctor : public ScalarLogicalBaseFunctor {
 public:
  ScalarLogicalGreaterFunctor() : ScalarLogicalBaseFunctor(/*op_name=*/"scalar_logical_greater") {}
};

// (scalar > x) = (x < scalar)
class ScalarLogicalGreater2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarLogicalLess(x, scalar);
  }
};

class ScalarLogicalGreaterEqualFunctor : public ScalarLogicalBaseFunctor {
 public:
  ScalarLogicalGreaterEqualFunctor()
      : ScalarLogicalBaseFunctor(/*op_name=*/"scalar_logical_greater_equal") {}
};

// (scalar >= x) = (x <= scalar)
class ScalarLogicalGreaterEqual2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarLogicalLessEqual(x, scalar);
  }
};

class ScalarLogicalLessFunctor : public ScalarLogicalBaseFunctor {
 public:
  ScalarLogicalLessFunctor() : ScalarLogicalBaseFunctor(/*op_name=*/"scalar_logical_less") {}
};

// (scalar < x) = (x > scalar)
class ScalarLogicalLess2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarLogicalGreater(x, scalar);
  }
};

class ScalarLogicalLessEqualFunctor : public ScalarLogicalBaseFunctor {
 public:
  ScalarLogicalLessEqualFunctor()
      : ScalarLogicalBaseFunctor(/*op_name=*/"scalar_logical_less_equal") {}
};

// (scalar <= x) = (x >= scalar)
class ScalarLogicalLessEqual2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarLogicalGreaterEqual(x, scalar);
  }
};

class ScalarLogicalAndFunctor : public ScalarLogicalBaseFunctor {
 public:
  ScalarLogicalAndFunctor() : ScalarLogicalBaseFunctor(/*op_name=*/"scalar_logical_and") {}
};

// (scalar && x) = (x && scalar)
class ScalarLogicalAnd2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarLogicalAnd(x, scalar);
  }
};

class ScalarLogicalOrFunctor : public ScalarLogicalBaseFunctor {
 public:
  ScalarLogicalOrFunctor() : ScalarLogicalBaseFunctor(/*op_name=*/"scalar_logical_or") {}
};

// (scalar || x) = (x || scalar)
class ScalarLogicalOr2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarLogicalOr(x, scalar);
  }
};

class ScalarLogicalXorFunctor : public ScalarLogicalBaseFunctor {
 public:
  ScalarLogicalXorFunctor() : ScalarLogicalBaseFunctor(/*op_name=*/"scalar_logical_xor") {}
};

// (scalar ^ x) = (x ^ scalar)
class ScalarLogicalXor2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarLogicalXor(x, scalar);
  }
};

class StandardDeviationFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input,
                           const Optional<std::vector<int32_t>>& dim,
                           const Optional<bool>& unbiased, const Optional<bool>& keepdim) const {
    const int32_t ndim = input->shape()->NumAxes();
    std::vector<int32_t> axis;
    axis.reserve(ndim);
    if (dim.has_value() == false) {
      for (int i = 0; i < ndim; ++i) { axis.emplace_back(i); }
    } else {
      std::vector<int32_t>& dims = *JUST(dim);
      CHECK_GE_OR_RETURN(ndim, dims.size())
          << "Dimension out of range, expected to be in range of [" << -ndim << ", " << ndim - 1
          << "], but got " << dims.size();
      axis.assign(dims.begin(), dims.end());
    }

    bool unbias = true;
    bool keepdims = false;
    if (unbiased.has_value()) { unbias = JUST(unbiased); }
    if (keepdim.has_value()) { keepdims = JUST(keepdim); }

    JUST(CheckAxis(axis, *input->shape()));
    if (axis.size() == 0) {
      return functional::Constant(*input->shape(), Scalar(0), *input->dtype(), NullOpt);
    }

    int32_t reduce_count = 1;
    if (axis.size() == 1) {
      reduce_count *= input->shape()->At(axis[0]);
    } else {
      for (int i = 0; i < axis.size(); ++i) { reduce_count *= input->shape()->At(axis[i]); }
    }

    bool is_double = input->dtype()->data_type() == DataType::kDouble;
    if (is_double) {
      const auto& sum = JUST(functional::ScalarDiv(
          JUST(functional::ReduceSum(JUST(functional::Square(input)), axis, keepdims)),
          Scalar((double)reduce_count)));
      const auto& square = JUST(functional::Square(JUST(functional::ScalarDiv(
          JUST(functional::ReduceSum(input, axis, keepdims)), Scalar((double)reduce_count)))));
      const auto& sub = JUST(functional::Sub(sum, square, /*inplace=*/false));
      if (unbias) {
        return functional::Sqrt(JUST(functional::ScalarMul(
            sub, Scalar((double)reduce_count / (double)(reduce_count - 1)), false)));
      }
      /*
      According to the std calculation formula,
      StandardDeviation = \sqrt {\frac {\sum _ {i=1}^ {N}X_ {i}^ {2}}{N}  -  \mu ^ {2}}
        = \sqrt{\frac {1}{N}\sum _ {i=1}^ {n} (x_ {i}-\mu )^ {2}  -\frac {1}{N}  N \mu ^ {2}}
        = \sqrt{\frac {\sum _ {i=1}^ {N}X_ {i}^ {2}}{N}  -  \mu ^ {2}}

      when we are in the last sqrt,
      if the value in the radical is <= 0, it may cause the result gradient to appear
      undefined(nan), which is normal. In this case, the gradient of ours and pytorch are different.
      Use abs(absolute value) can keep it consistent with pytorch:

      const auto& abs = JUST(functional::Abs(sub));
      return functional::Sqrt(abs);
      */
      // const auto& abs = JUST(functional::Abs(sub));
      // return functional::Sqrt(abs);
      return functional::Sqrt(sub);
    } else {
      //  If input tensor's dtype is float32, than cast it to double dtype,
      //  because float dtype has accuracy problem in float dtype, see:
      //  https://github.com/Oneflow-Inc/oneflow/issues/6526
      const auto& double_input = JUST(functional::Cast(input, DType::Double()));
      const auto& sum = JUST(functional::ScalarDiv(
          JUST(functional::ReduceSum(JUST(functional::Square(double_input)), axis, keepdims)),
          Scalar((double)reduce_count)));
      const auto& square = JUST(functional::Square(
          JUST(functional::ScalarDiv(JUST(functional::ReduceSum(double_input, axis, keepdims)),
                                     Scalar((double)reduce_count)))));
      const auto& sub = JUST(functional::Sub(sum, square, /*inplace=*/false));
      if (unbias) {
        return functional::Cast(
            JUST(functional::Sqrt(JUST(functional::ScalarMul(
                sub, Scalar((double)reduce_count / (double)(reduce_count - 1)), false)))),
            input->dtype());
      }
      return functional::Cast(JUST(functional::Sqrt(sub)), input->dtype());
    }
  }
};

class VarianceFunctor {
 public:
  VarianceFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("var").Input("input").Output("output").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input,
                           const Optional<std::vector<int32_t>>& dim,
                           const Optional<bool>& unbiased, const Optional<bool>& keepdim) const {
    if (!IsFloatingDataType(input->dtype()->data_type())) {
      return Error::RuntimeError() << "var only support floating point dtypes";
    }
    MutableAttrMap attrs;
    if (unbiased) { JUST(attrs.SetAttr<bool>("unbiased", JUST(unbiased))); }
    if (keepdim) { JUST(attrs.SetAttr<bool>("keepdim", JUST(keepdim))); }
    std::vector<int32_t> axis;
    const int ndim = input->shape()->NumAxes();
    axis.reserve(ndim);
    if (!dim) {
      for (int i = 0; i < ndim; i++) { axis.emplace_back(i); }
    } else {
      std::vector<int32_t>& dims = *JUST(dim);
      CHECK_GE_OR_RETURN(ndim, dims.size())
          << "Dimension out of range, expected to be in range of [" << -ndim << ", " << ndim - 1
          << "], but got " << dims.size();
      std::sort(dims.begin(), dims.end());
      axis.assign(dims.begin(), dims.end());
    }
    for (size_t i = 0; i < axis.size(); i++) {
      if (axis[i] < 0) { axis[i] += ndim; }
    }
    JUST(attrs.SetAttr<std::vector<int32_t>>("dim", axis));
    JUST(attrs.SetAttr<DataType>("dtype", input->dtype()->data_type()));

    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DotFunctor {
 public:
  DotFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dot").Input("x").Input("y").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& other) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, other});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};
class MovedimVecFunctor {
 public:
  MovedimVecFunctor() = default;
  static Maybe<void> CheckNoRepeat(const std::vector<int32_t>& perm, std::vector<int32_t>& perm_out,
                                   int32_t ndim, const std::string& desc) {
    std::vector<bool> is_used(ndim, false);
    FOR_RANGE(size_t, i, 0, perm.size()) {
      int32_t item = perm[i];
      if (item < 0) { item += ndim; }
      CHECK_GE_OR_RETURN(item, 0) << ", Dimension out of range (expected to be in range of ["
                                  << -ndim << ", " << ndim - 1 << "], but got " << perm[i] << ")";
      CHECK_LT_OR_RETURN(item, ndim)
          << ", Dimension out of range (expected to be in range of [" << -ndim << ", " << ndim - 1
          << "], but got " << perm[i] << ")";
      CHECK_EQ_OR_RETURN(is_used[item], false) << "repeated dim in " << desc;

      is_used[item] = true;
      perm_out[i] = item;
    }
    return Maybe<void>::Ok();
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::vector<int32_t>& source,
                           const std::vector<int32_t>& destination) const {
    int32_t ndim = input->shape()->NumAxes();
    int32_t dim = source.size();

    CHECK_EQ_OR_RETURN(source.size(), destination.size())
        << "movedim: Invalid source or destination dims: source (" << source.size()
        << " dims ) should contain the same number of dims as destination (" << destination.size()
        << " dims)";

    std::vector<int32_t> source_nopeat(dim);
    std::vector<int32_t> destination_nopeat(dim);

    JUST(CheckNoRepeat(source, source_nopeat, ndim, "source"));
    JUST(CheckNoRepeat(destination, destination_nopeat, ndim, "destination"));

    std::vector<int32_t> order(ndim);
    std::vector<int32_t> source_dims(ndim);
    std::vector<int32_t> destination_dims(ndim);

    std::iota(source_dims.begin(), source_dims.end(), 0);
    std::iota(destination_dims.begin(), destination_dims.end(), 0);

    FOR_RANGE(size_t, i, 0, dim) {
      order[destination_nopeat[i]] = source_nopeat[i];
      source_dims[source_nopeat[i]] = -1;
      destination_dims[destination_nopeat[i]] = -1;
    }

    std::remove(source_dims.begin(), source_dims.end(), -1);
    std::remove(destination_dims.begin(), destination_dims.end(), -1);

    int64_t rest_dim = ndim - dim;
    FOR_RANGE(size_t, i, 0, rest_dim) { order[destination_dims[i]] = source_dims[i]; }

    return Transpose(input, order);
  }
};

class MovedimIntFunctor {
 public:
  MovedimIntFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& source,
                           const int32_t& destination) const {
    std::vector<int32_t> src{source};
    std::vector<int32_t> dest{destination};
    return MovedimVec(input, src, dest);
  }
};

class TensorSplitVecFunctor {
 public:
  TensorSplitVecFunctor() = default;
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const std::vector<int32_t>& indices_or_sections,
                                const int32_t& dim) const {
    int32_t ndim = input->ndim();
    CHECK_OR_RETURN((dim >= -ndim) && (dim < ndim))
        << "Dimension out of range (expected to be in range of [" << -ndim << "," << ndim - 1
        << "], but got " << dim << ")";
    int32_t pos_dim = dim >= 0 ? dim : dim + ndim;

    std::vector<int64_t> start(ndim, 0);
    std::vector<int64_t> stop(ndim);
    std::vector<int64_t> step(ndim, 1);
    for (int32_t i = 0; i < ndim; i++) { stop[i] = input->dim(i); }

    int32_t num_indices = indices_or_sections.size();
    TensorTuple output(num_indices + 1);
    for (int32_t i = 0; i < num_indices; i++) {
      int32_t end_idx = indices_or_sections[i];
      stop[pos_dim] = end_idx;
      output[i] = JUST(Slice(input, start, stop, step));
      start[pos_dim] = end_idx;
    }
    stop[pos_dim] = input->shape()->At(ndim - 1);
    output[num_indices] = JUST(Slice(input, start, stop, step));

    return output;
  }
};

class TensorSplitIntFunctor {
 public:
  TensorSplitIntFunctor() = default;
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const int32_t& indices_or_sections, const int32_t& dim) const {
    int32_t ndim = input->ndim();
    CHECK_OR_RETURN((dim >= -ndim) && (dim < ndim))
        << "Dimension out of range (expected to be in range of [" << -ndim << "," << ndim - 1
        << "], but got " << dim << ")";
    CHECK_OR_RETURN(indices_or_sections > 0)
        << "number of sections must be larger than 0, got ," << indices_or_sections << ");";
    int32_t pos_dim = dim >= 0 ? dim : dim + ndim;

    const auto dim_size = input->dim(pos_dim);
    int64_t min_split_size = dim_size / indices_or_sections;
    int64_t num_splits_one_extra = dim_size % indices_or_sections;

    std::vector<int64_t> start(ndim, 0);
    std::vector<int64_t> stop(ndim);
    std::vector<int64_t> step(ndim, 1);
    for (int32_t i = 0; i < ndim; i++) { stop[i] = input->dim(i); }
    stop[pos_dim] = 0;

    TensorTuple output(indices_or_sections);
    for (int32_t i = 0; i < indices_or_sections; i++) {
      int64_t split_size = (i < num_splits_one_extra) ? (min_split_size + 1) : min_split_size;
      stop[pos_dim] += split_size;
      output[i] = JUST(Slice(input, start, stop, step));
      start[pos_dim] += split_size;
    }

    return output;
  }
};

class HsplitIntFunctor {
 public:
  HsplitIntFunctor() = default;
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const int32_t& indices_or_sections) const {
    int32_t ndim = input->ndim();
    CHECK_OR_RETURN(ndim >= 1)
        << "flow.hsplit requires a tensor with at least 1 dimension, but got a tensor with " << ndim
        << " dimensions!";
    CHECK_OR_RETURN(indices_or_sections > 0) << "indices_or_sections must greater than 0";
    int32_t dim = (ndim == 1) ? 0 : 1;
    CHECK_OR_RETURN(input->dim(dim) % indices_or_sections == 0)
        << "flow.hsplit attempted to split along dimension " << dim
        << ", but the size of the dimension " << input->shape()->At(dim)
        << " is not divisible by the split_size " << indices_or_sections << "!";
    return TensorSplitInt(input, indices_or_sections, dim);
  }
};

class HsplitVecFunctor {
 public:
  HsplitVecFunctor() = default;
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const std::vector<int32_t>& indices_or_sections) const {
    int32_t ndim = input->ndim();
    CHECK_OR_RETURN(ndim >= 1)
        << "flow.hsplit requires a tensor with at least 1 dimension, but got a tensor with " << ndim
        << " dimensions!";
    int32_t dim = (ndim == 1) ? 0 : 1;
    return TensorSplitVec(input, indices_or_sections, dim);
  }
};

class VsplitIntFunctor {
 public:
  VsplitIntFunctor() = default;
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const int32_t& indices_or_sections) const {
    int32_t ndim = input->ndim();
    CHECK_OR_RETURN(ndim >= 2)
        << "flow.vsplit requires a tensor with at least 2 dimension, but got a tensor with " << ndim
        << " dimensions!";
    CHECK_OR_RETURN(indices_or_sections > 0) << "indices_or_sections must greater than 0";
    CHECK_OR_RETURN(input->dim(0) % indices_or_sections == 0)
        << "flow.vsplit attempted to split along dimension " << 0
        << ", but the size of the dimension " << input->dim(0)
        << " is not divisible by the split_size " << indices_or_sections << "!";
    return TensorSplitInt(input, indices_or_sections, 0);
  }
};

class VsplitVecFunctor {
 public:
  VsplitVecFunctor() = default;
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& input,
                                const std::vector<int32_t>& indices_or_sections) const {
    int32_t ndim = input->shape()->NumAxes();
    CHECK_OR_RETURN(ndim >= 2)
        << "flow.vsplit requires a tensor with at least 1 dimension, but got a tensor with " << ndim
        << " dimensions!";
    return TensorSplitVec(input, indices_or_sections, 0);
  }
};

class ErfinvFunctor {
 public:
  ErfinvFunctor() { op_ = CHECK_JUST(one::OpBuilder("erfinv").Input("x").Output("y").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, {});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ErfinvInplaceFunctor {
 public:
  ErfinvInplaceFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("erfinv").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    JUST(CheckInplaceValid(x));
    std::shared_ptr<TensorTuple> outputs = std::make_shared<TensorTuple>(1);
    outputs->at(0) = x;
    JUST(OpInterpUtil::Dispatch(*op_, {x}, outputs.get(), {}));
    return outputs->at(0);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CumBaseFunctor {
 public:
  explicit CumBaseFunctor(std::string op_name) {
    op_ = CHECK_JUST(one::OpBuilder(op_name).Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, int64_t dim,
                           const Optional<Symbol<DType>>& dtype) const {
    auto ndim = input->ndim();
    if (dim < 0) { dim += ndim; }
    CHECK_OR_RETURN(dim >= 0 && dim < ndim)
        << "IndexError: Dimension out of range (expected to be in range of [" << -ndim << ","
        << ndim << " ) but got " << dim;

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("dim", dim));
    TensorProcessor tensor_processor;
    if (dtype) {
      JUST(tensor_processor.AddInputs({input}, JUST(dtype)).Apply());
    } else {
      JUST(tensor_processor.AddInputs({input}, DType::Int64()).Apply());
    }
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<Tensor>(*op_, input_tuple, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CumsumFunctor : public CumBaseFunctor {
 public:
  CumsumFunctor() : CumBaseFunctor("cumsum") {}
};

class CumProdFunctor : public CumBaseFunctor {
 public:
  CumProdFunctor() : CumBaseFunctor("cumprod") {}
};

class CumGradBaseFunctor {
 protected:
  std::shared_ptr<OpExpr> op_;
};

class CumsumGradFunctor : public CumGradBaseFunctor {
 public:
  CumsumGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("cumsum_grad").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, int64_t dim) const {
    // No need to check dim validation here, while CumsumFunctor handled already
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("dim", dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input}, attrs);
  }
};

class CumProdGradFunctor : public CumGradBaseFunctor {
 public:
  CumProdGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("cumprod_grad")
                         .Input("dy")
                         .Input("output")
                         .Input("input")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& x, int64_t dim) const {
    // No need to check dim validation here, while CumProdFunctor handled already
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("dim", dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, y, x}, attrs);
  }
};

// NOTE(Liang Depeng): The implementation of sumproduct_pair are mostly taken from pytorch.
//                     For more details pls refer to:
//                     https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Linear.cpp#L65

// sumproduct_pair computes `(left*right).sum(sumdims)` by means of permutation and
// batch matrix multiplication
// its main purpose is to provide a pairwise reduction for einsum
static Maybe<one::Tensor> sumproduct_pair(const std::shared_ptr<one::Tensor>& left_,
                                          const std::shared_ptr<one::Tensor>& right_,
                                          const std::vector<int32_t>& sum_dims_, bool keepdim) {
  // assumes that tensors have been pre-unsqueezed (so that all dimensions match - after
  // broadcasting) but makes no other assumptions on the order of dimensions
  CHECK_OR_RETURN(left_->ndim() == right_->ndim()) << "number of dimensions must match";
  if (sum_dims_.size() == 0) return functional::Mul(left_, right_);
  int64_t dim = left_->ndim();

  constexpr size_t dim_bitset_size = 64;
  CHECK_OR_RETURN(dim <= (int64_t)dim_bitset_size)
      << "only tensors with up to " << dim_bitset_size << " dims are supported";
  std::bitset<dim_bitset_size> sum_dims;
  for (int i = 0; i < sum_dims_.size(); ++i) {
    size_t d = sum_dims_[i];
    CHECK_OR_RETURN(!sum_dims[d]) << "dim " << d << " appears multiple times in the list of dims";
    sum_dims[d] = true;
  }

  // dimensions that will be part of the output (i.e. not summed over) in three vectors
  // dims in lro appear in left, right and output, similarly lo: left and output, ro: right and
  // output also the sizes are kept track of for reshaping
  std::vector<int32_t> lro, lo, ro;
  int32_t lro_size = 1, lo_size = 1, ro_size = 1, sum_size = 1;
  std::shared_ptr<one::Tensor> left = left_;
  std::shared_ptr<one::Tensor> right = right_;
  for (int i = 0; i < dim; ++i) {
    auto sl = left->shape()->At(i) > 1;
    auto sr = right->shape()->At(i) > 1;
    if (sum_dims[i]) {  // first dimensions that will be summed over after multiplication
      if (sl && sr) {   // dimensions nontrivially in both left and right must be of the same size
        CHECK_OR_RETURN(left->shape()->At(i) == right->shape()->At(i))
            << "non-broadcast dimensions must match";
        sum_size *= left->shape()->At(i);
      } else if (sl) {  // if it is only in one of left and right, we can sum right away
        left = JUST(functional::ReduceSum(left, {i}, true));
      } else if (sr) {
        right = JUST(functional::ReduceSum(right, {i}, true));
      }
    } else if (sl && sr) {  // now deal with dimensions  dimensions that will be in the output
      // dimensions nontrivially in both left and right must be of the same size
      CHECK_OR_RETURN(left->shape()->At(i) == right->shape()->At(i))
          << "non-broadcast dimensions must match";
      lro.push_back(i);
      lro_size *= left->shape()->At(i);
    } else if (sl) {  // keep track of dimensions appearing only once
      lo.push_back(i);
      lo_size *= left->shape()->At(i);
    } else {
      ro.push_back(i);
      ro_size *= right->shape()->At(i);
    }
  }

  // we now work with the following permutations / shapes.
  // the pipeline is permute inputs -> reshape inputs -> batch matrix mul -> reshape(view) output ->
  // permute output output: "lro, lo, 1-for-summed-dims, ro" with orgiginal shape dimensions left:
  // "lro, lo, summed" permuted with lpermutation and the three flattened right:  "lro, summed, ro"
  // permuted with rpermutation and the three flattened then the permuted output is a view of
  // bmm(left, right) finally, opermutation reverts the permutation to the original order of
  // dimensions
  std::vector<int32_t> out_size;
  for (auto& d : lro) out_size.push_back(left->shape()->At(d));
  for (auto& d : lo) out_size.push_back(left->shape()->At(d));
  for (auto& d : sum_dims_) {
    out_size.push_back(1);
    (void)(d);
  };  // avoid warining about not using d
  for (auto& d : ro) out_size.push_back(right->shape()->At(d));

  std::vector<int32_t> lpermutation(lro);
  lpermutation.insert(lpermutation.end(), lo.begin(), lo.end());
  lpermutation.insert(lpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  lpermutation.insert(lpermutation.end(), ro.begin(), ro.end());

  std::vector<int32_t> rpermutation(lro);
  rpermutation.insert(rpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  rpermutation.insert(rpermutation.end(), ro.begin(), ro.end());
  rpermutation.insert(rpermutation.end(), lo.begin(), lo.end());

  std::vector<int32_t> opermutation(lro.size() + lo.size() + sum_dims_.size() + ro.size(), -1);
  {
    int32_t i = 0;

    for (auto it = lro.cbegin(); it != lro.cend(); i++, it++) { opermutation[*it] = i; }
    for (auto it = lo.cbegin(); it != lo.cend(); i++, it++) { opermutation[*it] = i; }
    for (auto it = sum_dims_.cbegin(); it != sum_dims_.cend(); i++, it++) { opermutation[*it] = i; }
    for (auto it = ro.cbegin(); it != ro.cend(); i++, it++) { opermutation[*it] = i; }
  }

  // now we can execute the operations above
  left = JUST(functional::Permute(left, lpermutation));
  DimVector lsv(3);
  lsv[0] = lro_size;
  lsv[1] = lo_size;
  lsv[2] = sum_size;
  const Shape ls(lsv);

  left = JUST(functional::Reshape(left, ls));

  right = JUST(functional::Permute(right, rpermutation));
  DimVector rsv(3);
  rsv[0] = lro_size;
  rsv[1] = sum_size;
  rsv[2] = ro_size;
  const Shape rs(rsv);
  right = JUST(functional::Reshape(right, rs));

  std::shared_ptr<one::Tensor> result =
      JUST(functional::BatchMatMul(left, right, false, false, 1.0));
  DimVector osv(out_size.size());
  for (int i = 0; i < out_size.size(); ++i) { osv[i] = out_size[i]; }
  const Shape os(osv);
  // TODO(Liang Depeng): change reshape to veiw
  result = JUST(functional::Reshape(result, os));
  result = JUST(functional::Permute(result, opermutation));

  // finally squeeze summed dimensions if desired
  if (!keepdim) {
    auto sizes = result->shape()->dim_vec();
    for (int i = dim - 1; i >= 0; i--) {
      if (sum_dims[i]) { sizes.erase(sizes.begin() + i); }
    }
    // TODO(Liang Depeng): change reshape to veiw
    const Shape s(sizes);
    result = JUST(functional::Reshape(result, s));
  }
  return result;
}

namespace {

bool einsum_check_label(unsigned char label) { return std::isalpha(label); }

uint8_t einsum_label_to_index(unsigned char label) {
  constexpr uint8_t NUM_OF_LETTERS = 'z' - 'a' + 1;
  return std::isupper(label) ? label - 'A' : NUM_OF_LETTERS + (label - 'a');
}

unsigned char einsum_index_to_label(uint8_t index) {
  constexpr uint8_t NUM_OF_LETTERS = 'z' - 'a' + 1;
  return index < NUM_OF_LETTERS ? index + 'A' : index - NUM_OF_LETTERS + 'a';
}

}  // namespace

// NOTE(Liang Depeng): The implementation of EinSumFunctor are mostly taken from pytorch.
//                     For more details pls refer to:
//                     https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Linear.cpp#L190

// There are roughly three parts to compute einsum:
// 1. Parse equation to extract the labels for each input operand and output
// 2. Unsqueeze missing dimensions from input operands and permute to align them
// 3. Compute result by multiplying input operands and summing contraction
//    dimensions We do the last part by reducing to batch matmul.
class EinSumFunctor {
 public:
  EinSumFunctor() {}
  Maybe<Tensor> operator()(const std::string& equation, const one::TensorTuple& operands) const {
    CHECK_OR_RETURN(operands.size() > 0) << "einsum(): must provide at least one input tensor.";
    // NOTE(Liang Depeng): In order to better understand what einsum is doing,
    //                     the following comments will give a detailed explaination of
    //                     how the operands of equation "ik,jkl,il->ij" (bilinear)
    //                     are transformed during the computation.
    //                     Assume that the size of each operands "ik", "jkl" and "il" are
    //                     [2, 3], [4, 3, 5], [2, 5] respectively.

    // Code used to identify ELLIPSIS ("...")
    constexpr uint8_t ELLIPSIS = 52;

    // Find arrow (->) to split equation into lhs (input equations) and rhs (output equation)
    const auto arrow_pos = equation.find("->");
    const auto lhs = equation.substr(0, arrow_pos);

    const auto num_ops = operands.size();

    // Convert each input equations into indexes in range [0, 52] and store
    // them in op_labels for each operand along with ELLIPSIS if present.
    std::vector<std::vector<uint8_t>> op_labels(num_ops);
    // NOTE(Liang Depeng): Continue explaining the equation "ik,jkl,il->ij".
    //                     After running the following for loop, `op_labels` contains 3 vectors.
    //                     The contents of each vectors are:
    //                     op_labels[0]: [34('i'-'a'+26), 36('k'-'a'+26)]
    //                     op_labels[1]: [35('j'-'a'+26), 36('k'-'a'+26), 37('l'-'a'+26)]
    //                     op_labels[2]: [34('i'-'a'+26), 37('l'-'a'+26)]
    bool found_ell = false;
    std::size_t curr_op = 0;
    for (auto i = decltype(lhs.length()){0}; i < lhs.length(); ++i) {
      const unsigned char label = lhs[i];
      switch (label) {
        case ' ':
          // Ignore spaces
          break;

        case '.':
          // process ellipsis
          CHECK_OR_RETURN(
              // Only one ellipsis per operand can be given
              !found_ell)
              << "einsum(): found \'.\' for operand " << curr_op
              << " for which an ellipsis was already found";
          CHECK_OR_RETURN(
              // Ensure it's a valid ellipsis
              i + 2 < lhs.length() && lhs[++i] == '.' && lhs[++i] == '.')
              << "einsum(): found \'.\' for operand " << curr_op
              << " that is not part of any ellipsis";
          op_labels[curr_op].push_back(ELLIPSIS);
          found_ell = true;
          break;

        case ',':
          // Move onto next operand
          ++curr_op;
          CHECK_OR_RETURN(curr_op < num_ops)
              << "einsum(): fewer operands were provided than specified in the equation";
          found_ell = false;
          break;

        default:
          // Parse label
          CHECK_OR_RETURN(einsum_check_label(label))
              << "einsum(): invalid subscript given at index  " << i
              << " in the equation string, subscripts must be in [a-zA-Z]";
          op_labels[curr_op].push_back(einsum_label_to_index(label));
      }
    }

    CHECK_OR_RETURN(curr_op == num_ops - 1)
        << "einsum(): more operands were provided than specified in the equation";

    // Labels must be within [a-zA-Z].
    constexpr uint8_t TOTAL_LABELS = 52;
    std::vector<int32_t> label_count(TOTAL_LABELS, 0);

    // The maximum number of dimensions covered by any ellipsis, needed when
    // unsqueezing missing dimensions from operands to permute and broadcast
    int32_t ell_num_dim = 0;
    // NOTE(Liang Depeng): Continue explaining the equation "ik,jkl,il->ij".
    //                     After running the following for loop,
    //                     the none zero indexes of `label_count` are:
    //                     op_labels[34] = 2
    //                     op_labels[35] = 1
    //                     op_labels[36] = 2
    //                     op_labels[37] = 2
    //                     `ell_num_dim` equals to 0 because no ellipsis in equation

    // Compute label frequency and number of dimensions covered by ellipsis
    // We do this after parsing labels to make it more readable and simpler
    // to compute the number of dimensions covered by ellipsis.
    for (auto i = 0; i < num_ops; i++) {
      const auto operand = operands[i];
      const auto labels = op_labels[i];
      const int ndims = operand->ndim();
      int32_t nlabels = static_cast<int32_t>(labels.size());
      bool has_ellipsis = false;

      for (const auto& label : labels) {
        if (label == ELLIPSIS) {
          --nlabels;
          has_ellipsis = true;
          ell_num_dim = std::max(ell_num_dim, ndims - nlabels);
        } else {
          ++label_count[label];
        }
      }
      if (has_ellipsis) {
        CHECK_OR_RETURN(nlabels <= ndims)
            << "einsum() the number of subscripts in the equation (" << nlabels
            << ") is more than the number of dimensions (" << ndims << ") for operand " << i;
      } else {
        CHECK_OR_RETURN(nlabels == ndims)
            << "einsum(): the number of subscripts in the equation (" << nlabels
            << ") does not match the number of dimensions (" << ndims << ") for operand " << i
            << " and no ellipsis was given";
      }
    }

    // We want to align the dimensions of every input tensor to have
    // shape out_dims + sum_dims. For this, we create a mapping of label
    // to index into the permuted shape.
    std::vector<int32_t> label_perm_index(TOTAL_LABELS, -1);

    // Current index in the permuted shape
    int32_t perm_index = 0;

    // Start index of ellipsis dimensions in the permuted shape
    int32_t ell_index = 0;
    found_ell = false;

    // NOTE(Liang Depeng): Continue explaining the equation "ik,jkl,il->ij".
    //                     After running the following if-else code block,
    //                     the none -1 indexes of `label_perm_index` are:
    //                     label_perm_index[34] = 0
    //                     label_perm_index[35] = 1
    //                     `perm_index` equals to 2
    //                     `ell_index` equals to 0 because no ellipsis in equation
    //                     `found_ell` equals to false because no ellipsis in equation
    if (arrow_pos == std::string::npos) {
      // Implicit output is ellipsis (...) + labels seen only once
      perm_index = ell_num_dim;
      found_ell = true;
      for (auto label = 0; label < TOTAL_LABELS; label++) {
        if (label_count[label] == 1) { label_perm_index[label] = perm_index++; }
      }
    } else {
      // Parse explicit output
      const auto rhs = equation.substr(arrow_pos + 2);
      for (auto i = decltype(rhs.length()){0}; i < rhs.length(); ++i) {
        const unsigned char label = rhs[i];
        switch (label) {
          case ' ':
            // Ignore spaces
            break;

          case '.':
            // process ellipsis
            CHECK_OR_RETURN(
                // There can only be one ellipsis in the output
                !found_ell)
                << "einsum(): found \'.\' for output but an ellipsis (...) was already found";
            CHECK_OR_RETURN(
                // Ensure ellipsis is correct
                i + 2 < rhs.length() && rhs[++i] == '.' && rhs[++i] == '.')
            "einsum(): found \'.\' for output that is not part of any ellipsis (...)";
            ell_index = perm_index;
            perm_index += ell_num_dim;
            found_ell = true;
            break;

          default:
            CHECK_OR_RETURN(einsum_check_label(label))
                << "einsum(): invalid subscript given at index " << lhs.size() + 2 + i
                << " in the equation string, subscripts must be in [a-zA-Z]";
            const auto index = einsum_label_to_index(label);
            CHECK_OR_RETURN(
                // Ensure label appeared at least once for some input operand
                // and at most once for the output
                label_count[index] > 0 && label_perm_index[index] == -1)
                << "einsum(): output subscript " << label
                << (label_perm_index[index] > -1
                        ? " appears more than once in the output"
                        : " does not appear in the equation for any input operand");
            label_perm_index[index] = perm_index++;
        }
      }
    }

    // Save output size before adding contraction dims (dims to sum out)
    const int32_t out_size = perm_index;

    // If ellipsis is not part of the output, add to contraction dimensions
    if (!found_ell) {
      ell_index = perm_index;
      perm_index += ell_num_dim;
    }

    // NOTE(Liang Depeng): Continue explaining the equation "ik,jkl,il->ij".
    //                     After running the following foor loop,
    //                     the none -1 indexes of `label_perm_index` are:
    //                     label_perm_index[34] = 0 ('i')
    //                     label_perm_index[35] = 1 ('j')
    //                     label_perm_index[36] = 2 ('k')
    //                     label_perm_index[37] = 3 ('l')
    //                     `out_size` equals to 2
    //                     `perm_index` equals to 4

    // Add contraction labels (labels not present in output)
    for (auto label = 0; label < TOTAL_LABELS; label++) {
      if (label_count[label] > 0 && label_perm_index[label] == -1) {
        label_perm_index[label] = perm_index++;
      }
    }

    // Here we unsqueeze missing dimensions to make all operands have the same
    // number of dimensions. We take diagonals for repeated labels within the
    // same operand. Finally we permute the operands to align dimensions as
    // per the perm_out_index we computed above.
    TensorTuple permuted_operands;
    for (auto i = 0; i < num_ops; i++) {
      std::vector<int32_t> perm_shape(perm_index, -1);
      std::vector<int32_t> label_dim(TOTAL_LABELS, -1);
      std::shared_ptr<Tensor> operand = operands[i];
      const auto labels = op_labels[i];
      const auto original_sizes = operand->shape()->dim_vec();

      int32_t j = 0;
      for (const auto& label : labels) {
        if (label == ELLIPSIS) {
          // Add missing dimensions covered by the ellipsis
          const auto num_missing_dim = ell_num_dim - (original_sizes.size() - labels.size() + 1);
          for (auto k = 0; k < num_missing_dim; k++) {
            operand = JUST(functional::Unsqueeze(operand, j));
          }
          for (auto k = 0; k < ell_num_dim; k++) { perm_shape[ell_index + k] = j++; }
        } else if (label_dim[label] != -1) {
          // Repeated label, take diagonal
          const auto dim = label_dim[label];
          CHECK_OR_RETURN(operand->dim(j) == operand->dim(dim))
              << "einsum() subscript " << einsum_index_to_label(label)
              << " is repeated for operand " << i << " but the sizes don't match, "
              << operand->dim(j) << " != " << operand->dim(dim);

          operand = JUST(functional::Diagonal(operand, 0, dim, j));
          operand = JUST(functional::MovedimInt(operand, -1, dim));
        } else {
          // Lookup output index for label
          label_dim[label] = j;
          perm_shape[label_perm_index[label]] = j++;
        }
      }

      // Add dimensions for missing labels
      for (int32_t& index : perm_shape) {
        if (index == -1) {
          operand = JUST(functional::Unsqueeze(operand, -1));
          index = j++;
        }
      }
      permuted_operands.emplace_back(JUST(functional::Permute(operand, perm_shape)));

      // NOTE(Liang Depeng): Continue explaining the equation "ik,jkl,il->ij".
      //                     What is going on within this foor loop?
      //                     For operand "ik" size = [2, 3]:
      //                        `perm_shape` equals to [0, 2, 1, 3]
      //                        first unsqueeze "ik" to 4 dim, from [2, 3] to [2, 3, 1, 1]
      //                        then permute with `perm_shape`, from [2, 3, 1, 1] to [2, 1, 3, 1]
      //
      //                     For operand "jkl" size = [4, 3, 5]:
      //                        `perm_shape` equals to [3, 0, 1, 2]
      //                        first unsqueeze "jkl" to 4 dim, from [4, 3, 5] to [4, 3, 5, 1]
      //                        then permute with `perm_shape`, from [4, 3, 5, 1] to [1, 4, 3, 5]
      //
      //                     For operand "il" size = [2, 5]:
      //                        `perm_shape` equals to [0, 2, 3, 1]
      //                        first unsqueeze "ik" to 4 dim, from [2, 5] to [2, 5, 1, 1]
      //                        then permute with `perm_shape`, from [2, 5, 1, 1] to [2, 1, 1, 5]
    }

    // Check if operands broadcast and keep track of last operand with
    // dimension size != 1 for optimizing reductions
    std::vector<std::size_t> dim_last_op(perm_index, 0);
    bool has_zero_size_dim = false;
    // NOTE(Liang Depeng): Continue explaining the equation "ik,jkl,il->ij".
    //                     After running the following foor loop,
    //                     The contents of `dim_last_op` are:
    //                     dim_last_op[0] = 2
    //                     dim_last_op[1] = 1
    //                     dim_last_op[2] = 1
    //                     dim_last_op[3] = 2
    //                     `has_zero_size_dim` equals to false
    for (auto dim = 0; dim < perm_index; dim++) {
      auto broadcast_size = permuted_operands[0]->dim(dim);
      for (auto i = 1; i < num_ops; i++) {
        const auto dim_size = permuted_operands[i]->dim(dim);
        if (broadcast_size != dim_size && broadcast_size != 1 && dim_size != 1) {
          std::ostringstream msg;
          msg << "einsum(): operands do not broadcast with remapped shapes [original->remapped]:";
          for (auto j = 0; j < num_ops; j++) {
            msg << " " << operands[j]->shape()->DebugStr() << "->"
                << permuted_operands[j]->shape()->DebugStr();
          }
          CHECK_OR_RETURN(false) << msg.str();
        }
        if (dim_size != 1) {
          broadcast_size = dim_size;
          dim_last_op[dim] = i;
        }
      }
      has_zero_size_dim |= broadcast_size == 0;
    }

    // Compute result
    std::shared_ptr<Tensor> result = permuted_operands[0];

    // Fast path for when an operand has zero sized dim
    if (has_zero_size_dim) {
      DimVector out_shape(out_size);
      for (auto i = 0; i < out_size; i++) {
        out_shape[i] = permuted_operands[dim_last_op[i]]->dim(i);
      }

      const Shape shape(out_shape);
      return functional::Constant(shape, Scalar(0), *permuted_operands[0]->dtype(), NullOpt);
    }

    // Sum out or squeeze dimensions that are size 1 for all later operands
    int dim = out_size;
    for (int i = dim; i < perm_index; ++i, ++dim) {
      if (dim_last_op[i] == 0) {
        if (result->dim(dim) == 1) {
          std::vector<int32_t> dims = {dim--};
          result = JUST(functional::Squeeze(result, dims));
        } else {
          result = JUST(functional::ReduceSum(result, {dim--}, false));
        }
      }
    }

    for (auto i = 1; i < num_ops; i++) {
      auto operand = permuted_operands[i];
      std::vector<int32_t> sum_dims;

      // Sum out or squeeze dimensions that are size 1 for all later operands
      dim = out_size;
      for (int j = dim; j < perm_index; ++j, ++dim) {
        if (dim_last_op[j] < i) {
          std::vector<int32_t> dims = {dim--};
          operand = JUST(functional::Squeeze(operand, dims));
        } else if (dim_last_op[j] == i) {
          if (result->dim(dim) == 1) {
            operand = JUST(functional::ReduceSum(operand, {dim}, false));
            std::vector<int32_t> dims = {dim--};
            result = JUST(functional::Squeeze(result, dims));
          } else {
            sum_dims.push_back(dim);
          }
        }
      }

      // Multiply tensors and sum out dimensions in sum_dims
      if (sum_dims.empty()) {
        result = JUST(functional::Mul(result, operand));
      } else if (sum_dims.size() == result->shape()->NumAxes()) {
        auto flatten_result = JUST(functional::Flatten(result, 0, -1));
        auto flatten_operand = JUST(functional::Flatten(operand, 0, -1));
        result = JUST(functional::Dot(flatten_result, flatten_operand));
      } else {
        result = JUST(sumproduct_pair(result, operand, sum_dims, false));
      }

      // NOTE(Liang Depeng): Continue explaining the equation "ik,jkl,il->ij".
      //                     What is going on within this foor loop?
      //                     For iter i = 1:
      //                        result = permuted_operands[0], size = [2, 1, 3, 1]
      //                        operand = permuted_operands[1], size = [1, 4, 3, 5]
      //                        sum_dims = [2, ]
      //                        what happened in `sumproduct_pair` ?
      //                            result [2, 1, 3, 1] will be permuted to [2, 3, 1, 1] then
      //                                reshaped to [1, 2, 3]
      //                            operand [1, 4, 3, 5] will be permuted to [3, 4, 5, 1] then
      //                                reshape to [1, 3, 4 * 5]
      //                            perform batch_matmul(result, operand) => [1, 2, 4 * 5]
      //                            then reshape to [2, 1, 4, 5] then permute to
      //                            [2, 4, 1, 5], at last reshape to [2, 4, 5]
      //
      //                     For iter i = 2:
      //                        result, size = [2, 4, 5]
      //                        operand = permuted_operands[2], size = [2, 1, 1, 5]
      //                        squeeze operand from [2, 1, 1, 5] to [2, 1, 5]
      //                        sum_dims = [2,]
      //                        what happened in `sumproduct_pair` ?
      //                            result [2, 4, 5] will be permuted to [2, 4, 5] then
      //                                reshaped to [2, 4, 5]
      //                            operand [2, 1, 5] will be permuted to [2, 5, 1] then
      //                                reshape to [2, 5, 1]
      //                            perform batch_matmul(result, operand)=>[2, 4, 1]
      //                            then reshape to [2, 4, 1] then permute to [2, 4, 1]
      //                            at last reshape to [2, 4]
    }
    return result;
  }
};

}  // namespace impl

using namespace impl;

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<AddNFunctor>("Add");
  m.add_functor<ScalarAddFunctor, ScalarAdd2Functor>("ScalarAdd");
  m.add_functor<ScalarSubFunctor, ScalarSub2Functor>("ScalarSub");
  m.add_functor<ScalarMulFunctor, ScalarMul2Functor>("ScalarMul");
  m.add_functor<InplaceScalarMulFunctor>("InplaceScalarMul");
  m.add_functor<ScalarDivFunctor, ScalarDiv2Functor>("ScalarDiv");
  m.add_functor<InplaceScalarDivFunctor>("InplaceScalarDiv");
  m.add_functor<ScalarPowFunctor>("ScalarPow");
  m.add_functor<ScalarReversePowFunctor>("ScalarReversePow");
  m.add_functor<ScalarPowGradFunctor>("ScalarPowGrad");
  m.add_functor<ScalarReversePowGradFunctor>("ScalarReversePowGrad");
  m.add_functor<ReduceMaxFunctor>("ReduceMax");
  m.add_functor<MaxFunctor, Max2Functor>("Max");
  m.add_functor<ReduceMeanFunctor>("ReduceMean");
  m.add_functor<ReduceMinFunctor>("ReduceMin");
  m.add_functor<MinFunctor, Min2Functor>("Min");
  m.add_functor<AmaxFunctor>("Amax");
  m.add_functor<ReduceSumFunctor>("ReduceSum");
  m.add_functor<ReduceAllFunctor>("ReduceAll");
  m.add_functor<ReduceAnyFunctor>("ReduceAny");
  m.add_functor<ReduceProdFunctor>("ReduceProd");
  m.add_functor<ReduceMinDeviceStageFunctor>("ReduceMinDeviceStage");
  m.add_functor<ReduceMaxDeviceStageFunctor>("ReduceMaxDeviceStage");
  m.add_functor<ReduceMinGlobalStageFunctor>("ReduceMinGlobalStage");
  m.add_functor<ReduceMaxGlobalStageFunctor>("ReduceMaxGlobalStage");
  m.add_functor<ReduceMinDeviceStageGradFunctor>("ReduceMinDeviceStageGrad");
  m.add_functor<ReduceMaxDeviceStageGradFunctor>("ReduceMaxDeviceStageGrad");
  m.add_functor<ReduceMinGlobalStageGradFunctor>("ReduceMinGlobalStageGrad");
  m.add_functor<ReduceMaxGlobalStageGradFunctor>("ReduceMaxGlobalStageGrad");
  m.add_functor<TransposeFunctor>("Transpose");
  m.add_functor<Transpose2dimFunctor>("Transpose2dim");
  m.add_functor<TransposeFunctor>("Permute");
  m.add_functor<AsStridedFunctor>("AsStrided");
  m.add_functor<AsStridedGradFunctor>("AsStridedGrad");
  m.add_functor<Transpose2dimFunctor>("Swapaxes");
  m.add_functor<Transpose2dimFunctor>("Swapdims");
  m.add_functor<ArangeFunctor, Arange2Functor>("Arange");
  m.add_functor<ConsistentArangeFunctor, ConsistentArange2Functor>("ConsistentArange");
  m.add_functor<CastFunctor>("Cast");
  m.add_functor<ClampFunctor>("Clamp");
  m.add_functor<ClampInplaceFunctor>("ClampInplace");
  m.add_functor<ClipFunctor>("Clip");
  m.add_functor<ClipInplaceFunctor>("ClipInplace");
  m.add_functor<SqrtSquareSumFunctor>("SqrtSquareSum");
  m.add_functor<VectorNormFunctor, ScalarVectorNormFunctor>("VectorNorm");
  m.add_functor<ScalarMatrixNormFunctor, MatrixNormFunctor>("MatrixNorm");
  m.add_functor<NormFunctor, Norm2Functor>("Norm");
  m.add_functor<ScalarNormFunctor, ScalarNorm2Functor>("ScalarNorm");
  m.add_functor<ClampGradFunctor>("ClampGrad");
  m.add_functor<SelectFunctor>("Select");
  m.add_functor<SelectTopNFunctor>("SelectTopN");
  m.add_functor<MinimumFunctor>("Minimum");
  m.add_functor<MinimumFunctor>("Min");
  m.add_functor<MaximumFunctor>("Maximum");
  m.add_functor<MaximumFunctor>("Max");
  m.add_functor<ScalarFModFunctor>("ScalarFMod");
  m.add_functor<ScalarFloorDivFunctor>("ScalarFloorDiv");
  m.add_functor<ScalarLogicalEqualFunctor, ScalarLogicalEqual2Functor>("ScalarLogicalEqual");
  m.add_functor<ScalarLogicalNotEqualFunctor, ScalarLogicalNotEqual2Functor>(
      "ScalarLogicalNotEqual");
  m.add_functor<ScalarLogicalGreaterFunctor, ScalarLogicalGreater2Functor>("ScalarLogicalGreater");
  m.add_functor<ScalarLogicalGreaterEqualFunctor, ScalarLogicalGreaterEqual2Functor>(
      "ScalarLogicalGreaterEqual");
  m.add_functor<ScalarLogicalLessFunctor, ScalarLogicalLess2Functor>("ScalarLogicalLess");
  m.add_functor<ScalarLogicalLessEqualFunctor, ScalarLogicalLessEqual2Functor>(
      "ScalarLogicalLessEqual");
  m.add_functor<ScalarLogicalAndFunctor, ScalarLogicalAnd2Functor>("ScalarLogicalAnd");
  m.add_functor<ScalarLogicalOrFunctor, ScalarLogicalOr2Functor>("ScalarLogicalOr");
  m.add_functor<ScalarLogicalXorFunctor, ScalarLogicalXor2Functor>("ScalarLogicalXor");
  m.add_functor<StandardDeviationFunctor>("StandardDeviation");
  m.add_functor<VarianceFunctor>("Variance");
  m.add_functor<DotFunctor>("Dot");
  m.add_functor<MovedimVecFunctor>("MovedimVec");
  m.add_functor<MovedimIntFunctor>("MovedimInt");
  m.add_functor<TensorSplitVecFunctor>("TensorSplitVec");
  m.add_functor<TensorSplitIntFunctor>("TensorSplitInt");
  m.add_functor<HsplitIntFunctor>("HsplitInt");
  m.add_functor<HsplitVecFunctor>("HsplitVec");
  m.add_functor<VsplitIntFunctor>("VsplitInt");
  m.add_functor<VsplitVecFunctor>("VsplitVec");
  m.add_functor<ErfinvFunctor>("Erfinv");
  m.add_functor<ErfinvInplaceFunctor>("ErfinvInplace");
  m.add_functor<CumsumFunctor>("Cumsum");
  m.add_functor<CumsumGradFunctor>("CumsumGrad");
  m.add_functor<CumProdFunctor>("Cumprod");
  m.add_functor<CumProdGradFunctor>("CumprodGrad");
  m.add_functor<EinSumFunctor>("EinSum");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
