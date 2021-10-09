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
#include "oneflow/core/functional/functional_api.yaml.h"
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
        outputs.push_back(outs->at(0));
      } else {
        outputs.push_back(JUST(OpInterpUtil::Dispatch<Tensor>(*op_.at(size - 1), partial_inputs)));
      }
    }
    if (outputs.size() == 1) { return outputs.at(0); }
    return this->operator()(outputs, inplace);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> op_;
};

class ScalarAddFunctor {
 public:
  ScalarAddFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scalar_add").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar,
                           bool inplace) const {
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
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in ScalarAdd should be float or int.";
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

class ScalarAdd2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarAdd(x, scalar, /*inplace*/ false);
  }
};

class ScalarSubFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar,
                           bool inplace) const {
    return ScalarAdd(x, Scalar(-1) * scalar, inplace);
  }
};

class ScalarSub2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarAdd(JUST(ScalarMul(x, Scalar(-1))), scalar, /*inplace*/ false);
  }
};

class ScalarMulFunctor {
 public:
  ScalarMulFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scalar_mul").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    if (std::dynamic_pointer_cast<StaticZerosTensor>(x)) { return x; }
    MutableAttrMap attrs;
    TensorProcessor tensor_processor;
    Symbol<DType> lowest_dtype;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      // Only promote type to Float32 when tensor is Int type but scalar is float type.
      if (DType::priority_order[x->dtype()->data_type()]
          < DType::priority_order[DType::Float16()->data_type()]) {
        lowest_dtype = DType::Float();
      } else {
        lowest_dtype = x->dtype();
      }
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      lowest_dtype = x->dtype();
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in ScalarMul should be float or int.";
    }
    JUST(tensor_processor.AddInputs({x}, lowest_dtype).Apply());
    TensorTuple casted_vec = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<Tensor>(*op_, casted_vec, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ScalarMul2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return ScalarMul(x, scalar);
  }
};

class ScalarDivFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    return ScalarMul(x, Scalar(1.0) / scalar);
  }
};

class ScalarDiv2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& scalar, const std::shared_ptr<one::Tensor>& x) const {
    return functional::ScalarMul(JUST(functional::ReciprocalNoNan(x)), scalar);
  }
};

class ScalarPowFunctor {
 public:
  ScalarPowFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scalar_pow").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    MutableAttrMap attrs;
    TensorProcessor tensor_processor;
    Symbol<DType> lowest_dtype;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      // Only promote type to Float32 when tensor is Int type but scalar is float type.
      if (DType::priority_order[x->dtype()->data_type()]
          < DType::priority_order[DType::Float16()->data_type()]) {
        lowest_dtype = DType::Float();
      } else {
        lowest_dtype = x->dtype();
      }
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      lowest_dtype = x->dtype();
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in ScalarPow should be float or int.";
    }
    JUST(tensor_processor.AddInputs({x}, lowest_dtype).Apply());
    TensorTuple casted_vec = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<Tensor>(*op_, casted_vec, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
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

class ScalarFloorDivFunctor {
 public:
  ScalarFloorDivFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scalar_floordiv").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    if (std::dynamic_pointer_cast<StaticZerosTensor>(x)) { return x; }
    MutableAttrMap attrs;
    TensorProcessor tensor_processor;
    Symbol<DType> lowest_dtype;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      // Only promote type to Float32 when tensor is Int type but scalar is float type.
      if (DType::priority_order[x->dtype()->data_type()]
          < DType::priority_order[DType::Float16()->data_type()]) {
        lowest_dtype = DType::Float();
      } else {
        lowest_dtype = x->dtype();
      }
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      lowest_dtype = x->dtype();
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in ScalarFloorDiv should be float or int.";
    }
    JUST(tensor_processor.AddInputs({x}, lowest_dtype).Apply());
    TensorTuple casted_vec = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<Tensor>(*op_, casted_vec, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
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
    return functional::ScalarMul(sum, 1.0 / reduce_count);
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
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& permute) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("perm", permute));
    int32_t ndims = x->shape()->NumAxes();
    for (int i = 0; i < permute.size(); i++) {
      int32_t dim = permute.at(i);
      if (dim < 0) { dim += ndims; }
      CHECK_GE_OR_RETURN(dim, 0)
          << "IndexError: Dimension out of range (expected to be in range of [" << -ndims << ","
          << ndims << " ] but got " << ndims;
      CHECK_LT_OR_RETURN(dim, ndims)
          << "IndexError: Dimension out of range (expected to be in range of [" << -ndims << ","
          << ndims << " ] but got " << ndims;
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
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
    int32_t dim_0 = dim0;
    int32_t dim_1 = dim1;

    if (dim0 < 0) { dim_0 += ndim; }
    if (dim1 < 0) { dim_1 += ndim; }

    CHECK_OR_RETURN(dim_0 >= 0 && dim0 < ndim)
        << "Invalid dim0:" << dim_0 << " len(shape):" << ndim;
    CHECK_OR_RETURN(dim_1 >= 0 && dim1 < ndim)
        << "Invalid dim1:" << dim_1 << " len(shape):" << ndim;
    for (int32_t i = 0; i < ndim; ++i) { permute.push_back(i); }
    std::swap(permute[dim_0], permute[dim_1]);

    JUST(attrs.SetAttr<std::vector<int32_t>>("perm", permute));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ArangeFunctor {
 public:
  ArangeFunctor() { op_ = CHECK_JUST(one::OpBuilder("range").Output("out").Build()); }
  Maybe<Tensor> operator()(const Scalar& start, const Scalar& limit, const Scalar& delta,
                           const Symbol<DType>& dtype,
                           const Optional<Symbol<Device>>& device) const {
    MutableAttrMap attrs;
    const DataType range_dtype = dtype->data_type();
    JUST(attrs.SetAttr<DataType>("dtype", range_dtype));
    if (IsIntegralDataType(range_dtype)) {
      JUST(attrs.SetAttr<int64_t>("integer_start", JUST(start.As<int64_t>())));
      JUST(attrs.SetAttr<int64_t>("integer_limit", JUST(limit.As<int64_t>())));
      JUST(attrs.SetAttr<int64_t>("integer_delta", JUST(delta.As<int64_t>())));
    } else {
      JUST(attrs.SetAttr<double>("float_start", JUST(start.As<double>())));
      JUST(attrs.SetAttr<double>("float_limit", JUST(limit.As<double>())));
      JUST(attrs.SetAttr<double>("float_delta", JUST(delta.As<double>())));
    }
    OpExprInterpContext ctx(attrs);
    if (device) { ctx.device = JUST(device); }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {}, ctx);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class Arange2Functor {
 public:
  Maybe<Tensor> operator()(const Scalar& limit, const Symbol<DType>& dtype,
                           const Optional<Symbol<Device>>& device) const {
    return Arange(Scalar(0), limit, Scalar(1), dtype, device);
  }
};

class ConsistentArangeFunctor {
 public:
  ConsistentArangeFunctor() { op_ = CHECK_JUST(one::OpBuilder("range").Output("out").Build()); }
  Maybe<Tensor> operator()(const Scalar& start, const Scalar& limit, const Scalar& delta,
                           const Symbol<DType>& dtype, const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple) const {
    MutableAttrMap attrs;
    const DataType range_dtype = dtype->data_type();
    JUST(attrs.SetAttr<DataType>("dtype", range_dtype));
    if (IsIntegralDataType(range_dtype)) {
      JUST(attrs.SetAttr<int64_t>("integer_start", JUST(start.As<int64_t>())));
      JUST(attrs.SetAttr<int64_t>("integer_limit", JUST(limit.As<int64_t>())));
      JUST(attrs.SetAttr<int64_t>("integer_delta", JUST(delta.As<int64_t>())));
    } else {
      JUST(attrs.SetAttr<double>("float_start", JUST(start.As<double>())));
      JUST(attrs.SetAttr<double>("float_limit", JUST(limit.As<double>())));
      JUST(attrs.SetAttr<double>("float_delta", JUST(delta.As<double>())));
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
    return ConsistentArange(Scalar(0), limit, Scalar(1), dtype, placement, sbp_tuple);
  }
};

class ArgMaxFunctor : public UnaryFunctor {
 public:
  ArgMaxFunctor() { op_ = CHECK_JUST(one::OpBuilder("argmax").Input("in").Output("out").Build()); }
};

class ArgMinFunctor {
 public:
  ArgMinFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    return ArgMax(JUST(functional::Negative(x)));
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

class ClampFunctor {
 public:
  ClampFunctor() {
    clip_op_ = CHECK_JUST(one::OpBuilder("clip_by_scalar").Input("x").Output("y").Build());
    clip_min_op_ = CHECK_JUST(one::OpBuilder("clip_by_scalar_min").Input("x").Output("y").Build());
    clip_max_op_ = CHECK_JUST(one::OpBuilder("clip_by_scalar_max").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Optional<Scalar>& min,
                           const Optional<Scalar>& max) const {
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
    return OpInterpUtil::Dispatch<Tensor>(*op, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> clip_op_;
  std::shared_ptr<OpExpr> clip_min_op_;
  std::shared_ptr<OpExpr> clip_max_op_;
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
    const auto& output = JUST(OpInterpUtil::Dispatch<one::TensorTuple>(*op_, inputs, attr));
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

class ScalarFModFunctor {
 public:
  ScalarFModFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scalar_fmod").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    MutableAttrMap attrs;
    TensorProcessor tensor_processor;
    Symbol<DType> lowest_dtype;
    if (IsFloatingDataType(x->dtype()->data_type())) {
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
    } else if (IsIntegralDataType(x->dtype()->data_type())) {
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      lowest_dtype = x->dtype();
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in ScalarAdd should be float or int.";
    }
    JUST(tensor_processor.AddInputs({x}, lowest_dtype).Apply());
    TensorTuple casted_vec = JUST(tensor_processor.GetInputs());
    return OpInterpUtil::Dispatch<Tensor>(*op_, casted_vec, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
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
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in ScalarAdd should be float or int.";
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

}  // namespace impl

using namespace impl;

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<AddNFunctor>("Add");
  m.add_functor<ScalarAddFunctor, ScalarAdd2Functor>("ScalarAdd");
  m.add_functor<ScalarSubFunctor, ScalarSub2Functor>("ScalarSub");
  m.add_functor<ScalarMulFunctor, ScalarMul2Functor>("ScalarMul");
  m.add_functor<ScalarDivFunctor, ScalarDiv2Functor>("ScalarDiv");
  m.add_functor<ScalarPowFunctor>("ScalarPow");
  m.add_functor<ScalarPowGradFunctor>("ScalarPowGrad");
  m.add_functor<ReduceMaxFunctor>("ReduceMax");
  m.add_functor<ReduceMeanFunctor>("ReduceMean");
  m.add_functor<ReduceMinFunctor>("ReduceMin");
  m.add_functor<ReduceSumFunctor>("ReduceSum");
  m.add_functor<ReduceProdFunctor>("ReduceProd");
  m.add_functor<TransposeFunctor>("Transpose");
  m.add_functor<Transpose2dimFunctor>("Transpose2dim");
  m.add_functor<ArangeFunctor, Arange2Functor>("Arange");
  m.add_functor<ConsistentArangeFunctor, ConsistentArange2Functor>("ConsistentArange");
  m.add_functor<ArgMaxFunctor>("ArgMax");
  m.add_functor<ArgMinFunctor>("ArgMin");
  m.add_functor<CastFunctor>("Cast");
  m.add_functor<ClampFunctor>("Clamp");
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
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
