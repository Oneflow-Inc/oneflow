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
        outs->at(0) = partial_inputs.at(0);
        JUST(OpInterpUtil::Dispatch(*op_.at(size - 1), partial_inputs, outs.get()));
        outputs.emplace_back(outs->at(0));
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
      lowest_dtype = x->dtype();
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

class ScalarDivFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    return ScalarMul(x, Scalar(1.0) / scalar, /*inplace=*/false);
  }
};

class ScalarDiv2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return functional::ScalarMul(JUST(functional::ReciprocalNoNan(x)), scalar, /*inplace=*/false);
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
    if (reduce_count == 1) { return sum; }
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
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t dim0,
                           const int32_t dim1) const {
    MutableAttrMap attrs;
    const int64_t ndim = x->shape()->NumAxes();
    std::vector<int32_t> permute;
    permute.reserve(ndim);
    int32_t dim_0 = dim0;
    int32_t dim_1 = dim1;

    if (dim0 < 0) { dim_0 += ndim; }
    if (dim1 < 0) { dim_1 += ndim; }

    CHECK_OR_RETURN(dim_0 >= 0 && dim0 < ndim)
        << "Invalid dim0:" << dim_0 << " len(shape):" << ndim;
    CHECK_OR_RETURN(dim_1 >= 0 && dim1 < ndim)
        << "Invalid dim1:" << dim_1 << " len(shape):" << ndim;
    for (int32_t i = 0; i < ndim; ++i) { permute.emplace_back(i); }
    std::swap(permute[dim_0], permute[dim_1]);

    JUST(attrs.SetAttr<std::vector<int32_t>>("perm", permute));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SwapaxesFunctor {
 public:
  SwapaxesFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t dim0,
                           const int32_t dim1) const {
    const int64_t ndim = x->shape()->NumAxes();
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
    return Transpose2dim(x, dim0, dim1);
  }
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
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple) const {
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
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple) const {
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
        std::vector<int32_t> dim_column(1, 0);
        res = JUST(ReduceSum(JUST(ScalarLogicalNotEqual(x, 0)), dim_column, keepdim));
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
        dim_tmp.emplace_back(input_dim[i]);
      } else {
        dim_tmp.emplace_back(input_dim[i] + num_dims);
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

class SelectTopNFunctor {
 public:
  SelectTopNFunctor() { op_ = CHECK_JUST(one::SelectTopNOpExpr::New()); }

  Maybe<TensorTuple> operator()(const TensorTuple& inputs, int32_t n) const {
    MutableAttrMap attr;
    JUST(attr.SetAttr<int32_t>("top_n", n));
    std::vector<bool> require_grad(n);
    std::vector<bool> is_leaf(n);
    for (int i = 0; i < n; ++i) {
      is_leaf.at(i) = (inputs.at(i)->is_leaf());
      require_grad.at(i) = (inputs.at(i)->requires_grad());
    }
    const auto& output = JUST(OpInterpUtil::Dispatch<one::TensorTuple>(*op_, inputs, attr));
    for (int i = 0; i < n; ++i) {
      inputs.at(i)->set_is_leaf(is_leaf.at(i));
      JUST(inputs.at(i)->set_requires_grad(require_grad.at(i)));
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
    if (*x->shape() == *y->shape()) {
      return OpInterpUtil::Dispatch<Tensor>(*elementwise_minimum_op_, {x, y});
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*broadcast_minimum_op_, {x, y});
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
    if (*x->shape() == *y->shape()) {
      return OpInterpUtil::Dispatch<Tensor>(*elementwise_maximum_op_, {x, y});
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*broadcast_maximum_op_, {x, y});
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
    const DataType dtype = x->dtype()->data_type();
    MutableAttrMap attrs;

    if (IsFloatingDataType(dtype)) {
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
    } else if (IsIntegralDataType(dtype) || dtype == DataType::kUInt8) {
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in " << op_->op_type_name()
                                  << " should be float or int.";
    }

    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
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
      const auto& sub = JUST(functional::Sub(sum, square));
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
      const auto& sub = JUST(functional::Sub(sum, square));
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

class CumsumFunctor {
 public:
  CumsumFunctor() { op_ = CHECK_JUST(one::OpBuilder("cumsum").Input("x").Output("y").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, int64_t dim) const {
    auto ndim = input->ndim();
    if (dim < 0) { dim += ndim; }
    CHECK_OR_RETURN(dim >= 0 && dim < ndim)
        << "IndexError: Dimension out of range (expected to be in range of [" << -ndim << ","
        << ndim << " ) but got " << dim;

    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("dim", dim));
    TensorProcessor tensor_processor;
    JUST(tensor_processor.AddInputs({input}, DType::Int64()).Apply());
    TensorTuple input_tuple = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<Tensor>(*op_, input_tuple, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CumsumGradFunctor {
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

 private:
  std::shared_ptr<OpExpr> op_;
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
  m.add_functor<ScalarPowFunctor>("ScalarPow");
  m.add_functor<ScalarPowGradFunctor>("ScalarPowGrad");
  m.add_functor<ReduceMaxFunctor>("ReduceMax");
  m.add_functor<ReduceMeanFunctor>("ReduceMean");
  m.add_functor<ReduceMinFunctor>("ReduceMin");
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
  m.add_functor<TransposeFunctor>("Permute");
  m.add_functor<Transpose2dimFunctor>("Transpose2dim");
  m.add_functor<SwapaxesFunctor>("Swapaxes");
  m.add_functor<ArangeFunctor, Arange2Functor>("Arange");
  m.add_functor<ConsistentArangeFunctor, ConsistentArange2Functor>("ConsistentArange");
  m.add_functor<CastFunctor>("Cast");
  m.add_functor<ClampFunctor>("Clamp");
  m.add_functor<ClampInplaceFunctor>("ClampInplace");
  m.add_functor<ClampFunctor>("Clip");
  m.add_functor<ClampInplaceFunctor>("ClipInplace");
  m.add_functor<SqrtSquareSumFunctor>("SqrtSquareSum");
  m.add_functor<VectorNormFunctor, ScalarVectorNormFunctor>("VectorNorm");
  m.add_functor<ScalarMatrixNormFunctor, MatrixNormFunctor>("MatrixNorm");
  m.add_functor<NormFunctor, Norm2Functor>("Norm");
  m.add_functor<ScalarNormFunctor, ScalarNorm2Functor>("ScalarNorm");
  m.add_functor<ClampGradFunctor>("ClampGrad");
  m.add_functor<SelectTopNFunctor>("SelectTopN");
  m.add_functor<MinimumFunctor>("Minimum");
  m.add_functor<MaximumFunctor>("Maximum");
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
  m.add_functor<ErfinvFunctor>("Erfinv");
  m.add_functor<ErfinvInplaceFunctor>("ErfinvInplace");
  m.add_functor<CumsumFunctor>("Cumsum");
  m.add_functor<CumsumGradFunctor>("CumsumGrad");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
