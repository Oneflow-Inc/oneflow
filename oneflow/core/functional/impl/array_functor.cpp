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
#include "oneflow/core/functional/scalar.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class ConsistentConstantFunctor {
 public:
  ConsistentConstantFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("constant").Output("out").Build());
  }
  Maybe<Tensor> operator()(const Shape& shape, const Scalar& value, const Symbol<DType>& dtype,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
    if (IsIntegralDataType(dtype->data_type())) {
      JUST(attrs.SetAttr<bool>("is_floating_value", false));
      JUST(attrs.SetAttr<int64_t>("integer_value", JUST(value.As<int64_t>())));
    } else {
      JUST(attrs.SetAttr<bool>("is_floating_value", true));
      JUST(attrs.SetAttr<double>("floating_value", JUST(value.As<double>())));
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

class ConstantFunctor {
 public:
  ConstantFunctor() { op_ = CHECK_JUST(one::OpBuilder("constant").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Scalar& value, const Symbol<DType>& dtype,
                           const Optional<Symbol<Device>>& device) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
    if (IsIntegralDataType(dtype->data_type())) {
      JUST(attrs.SetAttr<bool>("is_floating_value", false));
      JUST(attrs.SetAttr<int64_t>("integer_value", JUST(value.As<int64_t>())));
    } else {
      JUST(attrs.SetAttr<bool>("is_floating_value", true));
      JUST(attrs.SetAttr<double>("floating_value", JUST(value.As<double>())));
    }
    if (device.has_value()) {
      Symbol<Device> device_symbol = JUST(device.value());
      return OpInterpUtil::Dispatch<Tensor>(*op_, {}, OpExprInterpContext(attrs, device_symbol));
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class EmptyFunctor {
 public:
  EmptyFunctor() { op_ = CHECK_JUST(one::OpBuilder("empty").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<DType>& dtype,
                           const Optional<Symbol<Device>>& device) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
    if (device.has_value()) {
      Symbol<Device> device_symbol = JUST(device.value());
      return OpInterpUtil::Dispatch<Tensor>(*op_, {}, OpExprInterpContext(attrs, device_symbol));
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_, {}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConsistentEmptyFunctor {
 public:
  ConsistentEmptyFunctor() { op_ = CHECK_JUST(one::OpBuilder("empty").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Symbol<DType>& dtype,
                           const Symbol<ParallelDesc>& placement,
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_tuple) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
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

class ZerosLikeFunctor : public UnaryFunctor {
 public:
  ZerosLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("zero_like").Input("like").Output("out").Build());
  }
};

class OnesLikeFunctor : public UnaryFunctor {
 public:
  OnesLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("ones_like").Input("like").Output("out").Build());
  }
};

class FlattenFunctor {
 public:
  FlattenFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("flatten").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t& start_dim,
                           const int32_t& end_dim) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("start_dim", start_dim));
    JUST(attrs.SetAttr<int32_t>("end_dim", end_dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class WhereFunctor {
 public:
  WhereFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("where").Input("condition").Input("x").Input("y").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition,
                           const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {condition, x, y});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class WhereScalarXFunctor {
 public:
  WhereScalarXFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("where_scalar_x").Input("condition").Input("y").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition, const Scalar& scalar,
                           const std::shared_ptr<one::Tensor>& y) const {
    MutableAttrMap attrs;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in Where shoule be float or int.";
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {condition, y}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class WhereScalarYFunctor {
 public:
  WhereScalarYFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("where_scalar_y").Input("condition").Input("x").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition,
                           const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    MutableAttrMap attrs;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in Where shoule be float or int.";
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {condition, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class WhereScalarXYFunctor {
 public:
  WhereScalarXYFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("where_scalar_xy").Input("condition").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& condition, const Scalar& x_scalar,
                           const Scalar& y_scalar) const {
    MutableAttrMap attrs;
    if (x_scalar.IsFloatingPoint() && y_scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("x_float_operand", JUST(x_scalar.As<double>())));
      JUST(attrs.SetAttr<double>("y_float_operand", JUST(y_scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_x_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_y_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_x_int_operand", false));
      JUST(attrs.SetAttr<bool>("has_y_int_operand", false));
    } else if (x_scalar.IsIntegral() && y_scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("x_int_operand", JUST(x_scalar.As<int64_t>())));
      JUST(attrs.SetAttr<int64_t>("y_int_operand", JUST(y_scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_x_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_y_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_x_int_operand", true));
      JUST(attrs.SetAttr<bool>("has_y_int_operand", true));
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "The scalar in Where shoule be float or int.";
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {condition}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ArgWhereFunctor {
 public:
  ArgWhereFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("argwhere").Input("input").Output("output").Output("output_size").Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const Symbol<DType>& dtype) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<DataType>("dtype", dtype->data_type()));
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastLikeFunctor {
 public:
  BroadcastLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_like").Input("x").Input("like").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& like,
                           const std::vector<int32_t>& broadcast_axes) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("broadcast_axes", broadcast_axes));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, like}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConcatFunctor {
 public:
  ConcatFunctor() {
    ops_.resize(kMaxInputCount);
    for (int n = 1; n < ops_.size(); ++n) {
      ops_[n] = CHECK_JUST(one::OpBuilder("concat").Input("in", n + 1).Output("out").Build());
    }
  }
  Maybe<Tensor> operator()(const TensorTuple& inputs, const int64_t& axis,
                           const int64_t& max_dim_size) const {
    CHECK_GE_OR_RETURN(inputs.size(), 2);
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("axis", axis));
    JUST(attrs.SetAttr<int64_t>("max_dim_size", max_dim_size));
    TensorTuple outputs;
    for (int i = 0; i < inputs.size(); i += kMaxInputCount) {
      size_t size = (i + kMaxInputCount) < inputs.size() ? kMaxInputCount : inputs.size() - i;
      TensorTuple partial_inputs(size);
      for (int j = 0; j < size; ++j) { partial_inputs[j] = inputs[i + j]; }
      outputs.push_back(
          JUST(OpInterpUtil::Dispatch<Tensor>(*ops_.at(size - 1), partial_inputs, attrs)));
    }
    if (outputs.size() == 1) { return outputs.at(0); }
    return this->operator()(outputs, axis, max_dim_size);
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> ops_;
};

class StackFunctor {
 public:
  StackFunctor() = default;
  Maybe<Tensor> operator()(const TensorTuple& inputs, const int64_t& dim) const {
    CHECK_GE_OR_RETURN(inputs.size(), 1) << "Needs one input at least.";
    int64_t ndims = inputs.at(0)->shape()->NumAxes();
    for (int i = 1; i < inputs.size(); ++i) {
      CHECK_EQ_OR_RETURN(inputs.at(i)->shape()->NumAxes(), ndims)
          << "The input dimensions are not equal.";
    }
    CHECK_OR_RETURN(dim >= 0 && dim <= ndims)
        << "The stack dim has to be between 0 and the input dimensions of " << ndims;
    TensorTuple expand_inputs(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) {
      expand_inputs[i] = JUST(ExpandDims(inputs.at(i), dim));
    }
    return Concat(expand_inputs, dim, inputs.size());
  }
};

class ExpandFunctor {
 public:
  ExpandFunctor() { op_ = CHECK_JUST(one::OpBuilder("expand").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Shape& shape) const {
    CHECK_GE_OR_RETURN(shape.NumAxes(), x->shape()->NumAxes())
        << "The desired expanded dims should not be less than the input dims.";
    std::vector<int32_t> in_shape(x->shape()->NumAxes());
    for (int i = 0; i < in_shape.size(); ++i) { in_shape[i] = x->shape()->At(i); }

    // calculate the original stride.
    std::vector<int32_t> original_stride(in_shape.size(), 1);
    for (int i = x->shape()->NumAxes() - 2; i >= 0; --i) {
      original_stride[i] = in_shape.at(i + 1) * original_stride.at(i + 1);
    }
    std::vector<int32_t> out_shape(shape.NumAxes());
    std::vector<int32_t> stride(shape.NumAxes());
    int shift = out_shape.size() - in_shape.size();
    for (int i = out_shape.size() - 1; i >= 0; --i) {
      int index = i - shift;
      if (index >= 0) {
        if (shape.At(i) == -1 || shape.At(i) == in_shape.at(index)) {
          out_shape[i] = in_shape.at(index);
          stride[i] = original_stride.at(index);
        } else {
          CHECK_OR_RETURN(shape.At(i) > 0 && in_shape.at(index) == 1)
              << "Invalid expand shape " << shape.ToString();
          out_shape[i] = shape.At(i);
          stride[i] = 0;
        }
      } else {
        CHECK_GT_OR_RETURN(shape.At(i), 0) << "Invalid expand shape " << shape.ToString();
        out_shape[i] = shape.At(i);
        if (shape.At(i) == 1 && i < out_shape.size() - 1) {
          stride[i] = stride.at(i + 1);
        } else {
          stride[i] = 0;
        }
      }
    }
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("in_shape", in_shape));
    JUST(attrs.SetAttr<std::vector<int32_t>>("out_shape", out_shape));
    JUST(attrs.SetAttr<std::vector<int32_t>>("stride", stride));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ExpandDimsFunctor {
 public:
  ExpandDimsFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("expand_dims").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t& axis) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("axis", axis));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GatherFunctor {
 public:
  GatherFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("gather").Input("in").Input("indices").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& indices, const int64_t& axis) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("axis", axis));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, indices}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimGatherFunctor {
 public:
  DimGatherFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("dim_gather").Input("input").Input("index").Output("output").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& indices, const int32_t& dim) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, indices}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterFunctor {
 public:
  DimScatterFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_update")
                         .Input("input")
                         .Input("index")
                         .Input("src")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index,
                           const std::shared_ptr<one::Tensor>& src) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index, src}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterAddFunctor {
 public:
  DimScatterAddFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_add")
                         .Input("input")
                         .Input("index")
                         .Input("src")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index,
                           const std::shared_ptr<one::Tensor>& src) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index, src}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterMulFunctor {
 public:
  DimScatterMulFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_mul")
                         .Input("input")
                         .Input("index")
                         .Input("src")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index,
                           const std::shared_ptr<one::Tensor>& src) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index, src}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterUpdateScalarFunctor {
 public:
  DimScatterUpdateScalarFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_update_scalar")
                         .Input("input")
                         .Input("index")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index, const Scalar& src) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    JUST(attrs.SetAttr<float>("src_scalar", JUST(src.As<float>())));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterAddScalarFunctor {
 public:
  DimScatterAddScalarFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_add_scalar")
                         .Input("input")
                         .Input("index")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index, const Scalar& src) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    JUST(attrs.SetAttr<float>("src_scalar", JUST(src.As<float>())));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DimScatterMulScalarFunctor {
 public:
  DimScatterMulScalarFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("dim_scatter_mul_scalar")
                         .Input("input")
                         .Input("index")
                         .Output("output")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input, const int32_t& dim,
                           const std::shared_ptr<one::Tensor>& index, const Scalar& src) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("dim", dim));
    JUST(attrs.SetAttr<float>("src_scalar", JUST(src.As<float>())));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {input, index}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ArgSortFunctor {
 public:
  ArgSortFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("arg_sort").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::string direction) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("direction", direction));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GatherNdFunctor {
 public:
  GatherNdFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("gather_nd").Input("params").Input("indices").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& params,
                           const std::shared_ptr<one::Tensor>& indices) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {params, indices});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ScatterNdFunctor {
 public:
  ScatterNdFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("scatter_nd").Input("indices").Input("updates").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& indices,
                           const std::shared_ptr<one::Tensor>& updates, const Shape& shape) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {indices, updates}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ScatterNdLikeFunctor {
 public:
  ScatterNdLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scatter_nd_like")
                         .Input("like")
                         .Input("updates")
                         .Input("indices")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& like,
                           const std::shared_ptr<one::Tensor>& updates,
                           const std::shared_ptr<one::Tensor>& indices) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {like, updates, indices});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ReshapeFunctor {
 public:
  ReshapeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("reshape").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Shape& shape) const {
    int need_infer_axis = -1;
    size_t count = 1;
    for (int i = 0; i < shape.NumAxes(); ++i) {
      if (shape.At(i) == -1) {
        CHECK_EQ_OR_RETURN(need_infer_axis, -1)
            << "Shape " << shape.ToString() << " has more than 1 axis that needs to be infered.";
        need_infer_axis = i;
      } else {
        count *= shape.At(i);
      }
    }
    size_t x_count = x->shape()->Count(0);
    MutableAttrMap attrs;
    if (need_infer_axis == -1) {
      CHECK_EQ_OR_RETURN(shape.Count(0), x_count);
      JUST(attrs.SetAttr<Shape>("shape", shape));
    } else {
      Shape infered_shape = shape;
      infered_shape.Set(need_infer_axis, x_count / count);
      CHECK_EQ_OR_RETURN(infered_shape.Count(0), x_count)
          << "Shape " << shape.ToString() << " is invalid for input of shape "
          << x->shape()->ToString();
      JUST(attrs.SetAttr<Shape>("shape", infered_shape));
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SliceBaseFunctor {
 public:
  SliceBaseFunctor() = default;
  virtual ~SliceBaseFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::vector<int64_t>& start,
                           const std::vector<int64_t>& stop,
                           const std::vector<int64_t>& step) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("start", start));
    JUST(attrs.SetAttr<std::vector<int64_t>>("stop", stop));
    JUST(attrs.SetAttr<std::vector<int64_t>>("step", step));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class SliceGradBaseFunctor {
 public:
  SliceGradBaseFunctor() = default;
  virtual ~SliceGradBaseFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& like,
                           const std::vector<int64_t>& start, const std::vector<int64_t>& stop,
                           const std::vector<int64_t>& step) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("start", start));
    JUST(attrs.SetAttr<std::vector<int64_t>>("stop", stop));
    JUST(attrs.SetAttr<std::vector<int64_t>>("step", step));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, like}, attrs);
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class SliceFunctor : public SliceBaseFunctor {
 public:
  SliceFunctor() { op_ = CHECK_JUST(one::OpBuilder("slice").Input("x").Output("y").Build()); }
};

class SliceGradFunctor : public SliceGradBaseFunctor {
 public:
  SliceGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("slice_grad").Input("dy").Input("like").Output("dx").Build());
  }
};

class NarrowFunctor {
 public:
  NarrowFunctor() { op_ = CHECK_JUST(one::OpBuilder("narrow").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in, const int64_t& dim,
                           const int64_t& start, const int64_t& length) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("dim", dim));
    JUST(attrs.SetAttr<int64_t>("start", start));
    JUST(attrs.SetAttr<int64_t>("length", length));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class NarrowGradFunctor {
 public:
  NarrowGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("narrow_grad").Input("dy").Input("like").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& like, const int64_t& dim,
                           const int64_t& start, const int64_t& length) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("dim", dim));
    JUST(attrs.SetAttr<int64_t>("start", start));
    JUST(attrs.SetAttr<int64_t>("length", length));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, like}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class LogicalSliceFunctor : public SliceBaseFunctor {
 public:
  LogicalSliceFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("logical_slice").Input("x").Output("y").Build());
  }
};

class LogicalSliceAssignFunctor {
 public:
  LogicalSliceAssignFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("logical_slice_assign").Input("ref").Input("value").Build());
  }
  Maybe<void> operator()(const std::shared_ptr<one::Tensor>& ref,
                         const std::shared_ptr<one::Tensor>& value,
                         const std::vector<int64_t>& start, const std::vector<int64_t>& stop,
                         const std::vector<int64_t>& step) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("start", start));
    JUST(attrs.SetAttr<std::vector<int64_t>>("stop", stop));
    JUST(attrs.SetAttr<std::vector<int64_t>>("step", step));
    JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_, {ref, value}, attrs));
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SliceUpdateFunctor {
 public:
  SliceUpdateFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("slice_update").Input("x").Input("update").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& update,
                           const std::vector<int64_t>& start, const std::vector<int64_t>& stop,
                           const std::vector<int64_t>& step) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int64_t>>("start", start));
    JUST(attrs.SetAttr<std::vector<int64_t>>("stop", stop));
    JUST(attrs.SetAttr<std::vector<int64_t>>("step", step));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, update}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class SqueezeFunctor {
 public:
  SqueezeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("squeeze").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& axes) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("axes", axes));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class CopyFunctor {
 public:
  CopyFunctor() { op_ = CHECK_JUST(one::OpBuilder("copy").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const std::string& device_type,
                           const int64_t& device_id) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::string>("device_type", device_type));
    JUST(attrs.SetAttr<int64_t>("device_id", device_id));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FlipFunctor {
 public:
  FlipFunctor() { op_ = CHECK_JUST(one::OpBuilder("flip").Input("x").Output("y").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& dims) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("dims", dims));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class FlipGradFunctor {
 public:
  FlipGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("flip_grad").Input("dy").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::vector<int32_t>& dims) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("dims", dims));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleFunctor {
 public:
  UpsampleFunctor() { op_ = CHECK_JUST(one::OpBuilder("upsample").Input("x").Output("y").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& height_scale,
                           const float& width_scale, const bool& align_corners,
                           const std::string& interpolation, const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("interpolation", interpolation));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleLinear1DFunctor {
 public:
  UpsampleLinear1DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_linear_1d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& scale_factor,
                           const bool& align_corners, const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("scale_factor", scale_factor));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleLinear1DGradFunctor {
 public:
  UpsampleLinear1DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_linear_1d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& scale_factor,
                           const bool& align_corners, const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("scale_factor", scale_factor));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest1DFunctor {
 public:
  UpsampleNearest1DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_nearest_1d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& scale_factor,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("scale_factor", scale_factor));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest1DGradFunctor {
 public:
  UpsampleNearest1DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_nearest_1d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& scale_factor,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("scale_factor", scale_factor));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest2DFunctor {
 public:
  UpsampleNearest2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_nearest_2d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& height_scale,
                           const float& width_scale, const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest2DGradFunctor {
 public:
  UpsampleNearest2DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_nearest_2d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& height_scale,
                           const float& width_scale, const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleBilinear2DFunctor {
 public:
  UpsampleBilinear2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_bilinear_2d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& height_scale,
                           const float& width_scale, const bool& align_corners,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleBilinear2DGradFunctor {
 public:
  UpsampleBilinear2DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_bilinear_2d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& height_scale,
                           const float& width_scale, const bool& align_corners,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleBicubic2DFunctor {
 public:
  UpsampleBicubic2DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_bicubic_2d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& height_scale,
                           const float& width_scale, const bool& align_corners,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleBicubic2DGradFunctor {
 public:
  UpsampleBicubic2DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_bicubic_2d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& height_scale,
                           const float& width_scale, const bool& align_corners,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest3DFunctor {
 public:
  UpsampleNearest3DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_nearest_3d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& depth_scale,
                           const float& height_scale, const float& width_scale,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("depth_scale", depth_scale));
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleNearest3DGradFunctor {
 public:
  UpsampleNearest3DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_nearest_3d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& depth_scale,
                           const float& height_scale, const float& width_scale,
                           const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("depth_scale", depth_scale));
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleTrilinear3DFunctor {
 public:
  UpsampleTrilinear3DFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("upsample_trilinear_3d").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& depth_scale,
                           const float& height_scale, const float& width_scale,
                           const bool& align_corners, const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("depth_scale", depth_scale));
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UpsampleTrilinear3DGradFunctor {
 public:
  UpsampleTrilinear3DGradFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("upsample_trilinear_3d_grad").Input("dy").Input("x").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const float& depth_scale,
                           const float& height_scale, const float& width_scale,
                           const bool& align_corners, const std::string& data_format) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("depth_scale", depth_scale));
    JUST(attrs.SetAttr<float>("height_scale", height_scale));
    JUST(attrs.SetAttr<float>("width_scale", width_scale));
    JUST(attrs.SetAttr<bool>("align_corners", align_corners));
    JUST(attrs.SetAttr<std::string>("data_format", data_format));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class UnsortedSegmentSumLikeFunctor {
 public:
  UnsortedSegmentSumLikeFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("unsorted_segment_sum_like")
                         .Input("data")
                         .Input("segment_ids")
                         .Input("like")
                         .Output("out")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& segment_ids,
                           const std::shared_ptr<one::Tensor>& like, const int64_t& axis) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("axis", axis));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, segment_ids, like}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TrilFunctor {
 public:
  TrilFunctor() { op_ = CHECK_JUST(one::OpBuilder("tril").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& diagonal) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("diagonal", diagonal));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TriuFunctor {
 public:
  TriuFunctor() { op_ = CHECK_JUST(one::OpBuilder("triu").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& diagonal) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int64_t>("diagonal", diagonal));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DiagFunctor {
 public:
  DiagFunctor() { op_ = CHECK_JUST(one::OpBuilder("diag").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const int32_t& diagonal) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("diagonal", diagonal));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DiagGradFunctor {
 public:
  DiagGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("diag_grad").Input("dy").Input("in").Output("dx").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dy,
                           const std::shared_ptr<one::Tensor>& x, const int32_t& diagonal) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("diagonal", diagonal));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dy, x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class TensorGetItemFunctor {
 public:
  TensorGetItemFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const TensorIndex& index) const {
    int64_t ndims = x->shape()->NumAxes();
    std::vector<detail::Slice> slice_indices;
    TensorTuple tensor_indices;
    std::vector<int64_t> target_dims;

    JUST(PrepareSliceIndices(index, *(x->shape()), &slice_indices, &tensor_indices, &target_dims));
    CHECK_EQ_OR_RETURN(slice_indices.size(), ndims) << "Failed to prepare slice indices.";
    Shape target_shape(DimVector(target_dims.begin(), target_dims.end()));

    std::vector<int64_t> start(ndims), end(ndims), step(ndims);
    for (int i = 0; i < ndims; ++i) {
      const auto& slice = slice_indices.at(i);
      start[i] = slice.start();
      end[i] = slice.end();
      step[i] = slice.step();
    }
    bool is_identity = [&]() {
      if (target_shape.NumAxes() == 0) { return false; }
      for (int i = 0; i < ndims; ++i) {
        if (start[i] != 0 || end[i] != x->shape()->At(i) || step[i] != 1) { return false; }
      }
      return true;
    }();
    std::shared_ptr<one::Tensor> result;
    if (is_identity) {
      result = JUST(Copy(x, JUST(x->device())->type(), JUST(x->device())->device_id()));
    } else {
      result = JUST(Slice(x, start, end, step));
    }

    Shape shape(DimVector(target_dims.begin(), target_dims.end()));
    if (shape.NumAxes() != 0 && shape != *(result->shape())) {
      result = JUST(Reshape(result, shape));
    }
    if (!tensor_indices.empty()) { result = JUST(ApplyAdvancedIndexing(result, tensor_indices)); }
    return result;
  }
};

class TensorSetItemFunctor {
 public:
  TensorSetItemFunctor() {}
  Maybe<void> operator()(const std::shared_ptr<one::Tensor>& x, const TensorIndex& index,
                         const std::shared_ptr<one::Tensor>& value) const {
    int64_t ndims = x->shape()->NumAxes();
    std::vector<detail::Slice> slice_indices;
    TensorTuple tensor_indices;
    std::vector<int64_t> target_dims;

    JUST(PrepareSliceIndices(index, *(x->shape()), &slice_indices, &tensor_indices, &target_dims));
    CHECK_EQ_OR_RETURN(slice_indices.size(), ndims) << "Failed to prepare slice indices.";
    CHECK_EQ_OR_RETURN(tensor_indices.size(), 0)
        << "Advanced indexing is not support for tensor setitem currently, please use basic "
           "indexing instead.";
    Shape target_shape(DimVector(target_dims.begin(), target_dims.end()));
    if (target_shape.Count(0) == 0) { return Maybe<void>::Ok(); }

    const auto& value_shape = value->shape();
    bool matched = [&]() {
      for (int i = 0; i < value_shape->NumAxes() - target_shape.NumAxes(); ++i) {
        if (value_shape->At(i) != 1) { return false; }
      }
      return true;
    }();
    CHECK_OR_RETURN(matched) << "The tensor size mismatch. Target sizes: "
                             << target_shape.ToString()
                             << ", value sizes: " << value_shape->ToString();
    std::shared_ptr<one::Tensor> value_tensor(value);
    if (target_shape.NumAxes() != 0 &&  // NOLINT
        /*need_expand=*/value_shape->Count(0) != target_shape.Count(0)) {
      // Remove the beginning redundant 1-dimensions.
      if (value_shape->NumAxes() > target_shape.NumAxes()) {
        int64_t start_axis = value_shape->NumAxes() - target_shape.NumAxes();
        const auto& shape = JUST(value_shape->Slice(start_axis, value_shape->NumAxes()));
        value_tensor = JUST(Reshape(value, *shape));
      }
      value_tensor = JUST(Expand(value_tensor, target_shape));
    }

    std::vector<int64_t> start(ndims), end(ndims), step(ndims);
    DimVector slice_dims(ndims);
    for (int i = 0; i < ndims; ++i) {
      const auto& slice = slice_indices.at(i);
      start[i] = slice.start();
      end[i] = slice.end();
      step[i] = slice.step();
      slice_dims[i] = (end[i] - start[i] + step[i] - 1) / step[i];
    }
    Shape slice_shape(slice_dims);
    if (slice_shape != *(value_tensor->shape())) {
      value_tensor = JUST(Reshape(value_tensor, slice_shape));
    }
    JUST(LogicalSliceAssign(x, value_tensor, start, end, step));
    return Maybe<void>::Ok();
  }
};

class CastLikeFunctor {
 public:
  CastLikeFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("cast_like").Input("in").Input("dtype_like").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& like) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, like});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ElementwiseMinimumGradFunctor {
 public:
  ElementwiseMinimumGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("elementwise_minimum_backward")
                         .Input("dz")
                         .Input("x")
                         .Input("y")
                         .Output("dx")
                         .Output("dy")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dz,
                                const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {dz, x, y});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ElementwiseMaximumGradFunctor {
 public:
  ElementwiseMaximumGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("elementwise_maximum_backward")
                         .Input("dz")
                         .Input("x")
                         .Input("y")
                         .Output("dx")
                         .Output("dy")
                         .Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& dz,
                                const std::shared_ptr<one::Tensor>& x,
                                const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<TensorTuple>(*op_, {dz, x, y});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DivGradFunctor {
 public:
  DivGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_div_grad")
                         .Input("dz")
                         .Input("z")
                         .Input("y")
                         .Output("dy")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dz,
                           const std::shared_ptr<one::Tensor>& z,
                           const std::shared_ptr<one::Tensor>& y) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dz, z, y});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastPowXGradFunctor {
 public:
  BroadcastPowXGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_pow_x_grad")
                         .Input("dz")
                         .Input("x")
                         .Input("y")
                         .Input("z")
                         .Output("dx")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dz,
                           const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& z) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dz, x, y, z});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastPowYGradFunctor {
 public:
  BroadcastPowYGradFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("broadcast_pow_y_grad")
                         .Input("dz")
                         .Input("x")
                         .Input("y")
                         .Input("z")
                         .Output("dy")
                         .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& dz,
                           const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& y,
                           const std::shared_ptr<one::Tensor>& z) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {dz, x, y, z});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class IdentityFunctor {
 public:
  IdentityFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("identity").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ReduceSumLikeFunctor {
 public:
  ReduceSumLikeFunctor() {
    op_ =
        CHECK_JUST(one::OpBuilder("reduce_sum_like").Input("x").Input("like").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::shared_ptr<one::Tensor>& like,
                           const std::vector<int32_t>& axis) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<std::vector<int32_t>>("axis", axis));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, like}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class BroadcastReduceSumLikeFunctor {
 public:
  BroadcastReduceSumLikeFunctor() {}
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input,
                           const std::shared_ptr<Tensor>& like) const {
    const auto& in_shape = *(input->shape());
    const auto& like_shape = *(like->shape());
    if (in_shape != like_shape) {
      const Shape& left_extended_shape =
          CreateLeftExtendedShape(ShapeView(like_shape), in_shape.NumAxes());
      if (in_shape == left_extended_shape) {
        return JUST(ReshapeLike(input, like));
      } else {
        const AxisVector& broadcast_axis_vec = left_extended_shape.Axes4BroadcastTo(in_shape);
        return JUST(ReduceSumLike(
            input, like,
            std::vector<int32_t>{broadcast_axis_vec.begin(), broadcast_axis_vec.end()}));
      }
    }
    return JUST(Identity(input));
  }
};

class SplitFunctor {
 public:
  SplitFunctor() {}
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x, const int64_t& split_size,
                                const int64_t& dim) const {
    CHECK_GE_OR_RETURN(split_size, 0)
        << "split expects split_size be non-negative, but got split_size=" << split_size;
    int64_t dim_size = x->shape()->At(dim);
    int64_t num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
    TensorTuple splits(num_splits);
    int64_t last_split_size = split_size - (split_size * num_splits - dim_size);
    for (int i = 0; i < num_splits; ++i) {
      int64_t length = i < num_splits - 1 ? split_size : last_split_size;
      splits[i] = JUST(Narrow(x, dim, i * split_size, length));
    }
    return splits;
  }
};

class SplitWithSizeFunctor {
 public:
  SplitWithSizeFunctor() {}
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const std::vector<int64_t>& split_sizes, const int64_t& dim) const {
    int64_t dim_size = x->shape()->At(dim);
    int64_t num_splits = split_sizes.size();
    TensorTuple splits(num_splits);
    int64_t start_idx = 0;
    for (int i = 0; i < num_splits; ++i) {
      int64_t length = split_sizes[i];
      CHECK_GE_OR_RETURN(length, 0) << "split_with_sizes expects split_sizes have only "
                                       "non-negative entries, but split_sizes["
                                    << i << "] = " << length;
      splits[i] = JUST(Narrow(x, dim, start_idx, length));
      start_idx += length;
    }
    CHECK_EQ_OR_RETURN(start_idx, dim_size)
        << "split_with_sizes expects split_sizes to sum exactly to " << dim_size
        << " (input tensor's size at dimension " << dim << "), "
        << "but got sum(split_sizes)=" << start_idx;
    return splits;
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ConsistentConstantFunctor>("ConsistentConstant");
  m.add_functor<impl::ConstantFunctor>("Constant");
  m.add_functor<impl::ConsistentEmptyFunctor>("ConsistentEmpty");
  m.add_functor<impl::EmptyFunctor>("Empty");
  m.add_functor<impl::ZerosLikeFunctor>("ZerosLike");
  m.add_functor<impl::OnesLikeFunctor>("OnesLike");
  m.add_functor<impl::FlattenFunctor>("Flatten");
  m.add_functor<impl::WhereFunctor>("Where");
  m.add_functor<impl::WhereScalarXFunctor>("WhereScalarX");
  m.add_functor<impl::WhereScalarYFunctor>("WhereScalarY");
  m.add_functor<impl::WhereScalarXYFunctor>("WhereScalarXY");
  m.add_functor<impl::ArgWhereFunctor>("ArgWhere");
  m.add_functor<impl::BroadcastLikeFunctor>("BroadcastLike");
  m.add_functor<impl::ConcatFunctor>("Concat");
  m.add_functor<impl::StackFunctor>("Stack");
  m.add_functor<impl::ExpandFunctor>("Expand");
  m.add_functor<impl::ExpandDimsFunctor>("ExpandDims");
  m.add_functor<impl::GatherFunctor>("Gather");
  m.add_functor<impl::DimGatherFunctor>("DimGather");
  m.add_functor<impl::ArgSortFunctor>("ArgSort");
  m.add_functor<impl::GatherNdFunctor>("GatherNd");
  m.add_functor<impl::ScatterNdFunctor>("ScatterNd");
  m.add_functor<impl::ScatterNdLikeFunctor>("ScatterNdLike");
  m.add_functor<impl::ReshapeFunctor>("Reshape");
  m.add_functor<impl::SliceFunctor>("Slice");
  m.add_functor<impl::SliceGradFunctor>("SliceGrad");
  m.add_functor<impl::NarrowFunctor>("Narrow");
  m.add_functor<impl::NarrowGradFunctor>("NarrowGrad");
  m.add_functor<impl::LogicalSliceAssignFunctor>("LogicalSliceAssign");
  m.add_functor<impl::LogicalSliceFunctor>("LogicalSlice");
  m.add_functor<impl::SliceUpdateFunctor>("SliceUpdate");
  m.add_functor<impl::SqueezeFunctor>("Squeeze");
  m.add_functor<impl::CopyFunctor>("Copy");
  m.add_functor<impl::FlipFunctor>("Flip");
  m.add_functor<impl::FlipGradFunctor>("FlipGrad");
  m.add_functor<impl::UpsampleFunctor>("Upsample");
  m.add_functor<impl::UpsampleNearest2DFunctor>("UpsampleNearest2D");
  m.add_functor<impl::UpsampleNearest2DGradFunctor>("UpsampleNearest2DGrad");
  m.add_functor<impl::UpsampleBilinear2DFunctor>("UpsampleBilinear2D");
  m.add_functor<impl::UpsampleBilinear2DGradFunctor>("UpsampleBilinear2DGrad");
  m.add_functor<impl::UpsampleLinear1DFunctor>("UpsampleLinear1D");
  m.add_functor<impl::UpsampleLinear1DGradFunctor>("UpsampleLinear1DGrad");
  m.add_functor<impl::UpsampleNearest1DFunctor>("UpsampleNearest1D");
  m.add_functor<impl::UpsampleNearest1DGradFunctor>("UpsampleNearest1DGrad");
  m.add_functor<impl::UpsampleBicubic2DFunctor>("UpsampleBicubic2D");
  m.add_functor<impl::UpsampleBicubic2DGradFunctor>("UpsampleBicubic2DGrad");
  m.add_functor<impl::UpsampleNearest3DFunctor>("UpsampleNearest3D");
  m.add_functor<impl::UpsampleNearest3DGradFunctor>("UpsampleNearest3DGrad");
  m.add_functor<impl::UpsampleTrilinear3DFunctor>("UpsampleTrilinear3D");
  m.add_functor<impl::UpsampleTrilinear3DGradFunctor>("UpsampleTrilinear3DGrad");
  m.add_functor<impl::UnsortedSegmentSumLikeFunctor>("UnsortedSegmentSumLike");
  m.add_functor<impl::TrilFunctor>("Tril");
  m.add_functor<impl::TriuFunctor>("Triu");
  m.add_functor<impl::DiagFunctor>("Diag");
  m.add_functor<impl::DiagGradFunctor>("DiagGrad");
  m.add_functor<impl::TensorGetItemFunctor>("TensorGetItem");
  m.add_functor<impl::DimScatterFunctor>("DimScatter");
  m.add_functor<impl::DimScatterAddFunctor>("DimScatterAdd");
  m.add_functor<impl::DimScatterMulFunctor>("DimScatterMul");
  m.add_functor<impl::DimScatterUpdateScalarFunctor>("DimScatterUpdateScalar");
  m.add_functor<impl::DimScatterAddScalarFunctor>("DimScatterAddScalar");
  m.add_functor<impl::DimScatterMulScalarFunctor>("DimScatterMulScalar");
  m.add_functor<impl::TensorSetItemFunctor>("TensorSetItem");
  m.add_functor<impl::CastLikeFunctor>("CastLike");
  m.add_functor<impl::ElementwiseMinimumGradFunctor>("ElementwiseMinGrad");
  m.add_functor<impl::ElementwiseMaximumGradFunctor>("ElementwiseMaxGrad");
  m.add_functor<impl::BroadcastPowXGradFunctor>("BroadcastPowXGrad");
  m.add_functor<impl::BroadcastPowYGradFunctor>("BroadcastPowYGrad");
  m.add_functor<impl::DivGradFunctor>("DivGrad");
  m.add_functor<impl::IdentityFunctor>("Identity");
  m.add_functor<impl::ReduceSumLikeFunctor>("ReduceSumLike");
  m.add_functor<impl::BroadcastReduceSumLikeFunctor>("BroadcastReduceSumLike");
  m.add_functor<impl::SplitFunctor>("Split");
  m.add_functor<impl::SplitWithSizeFunctor>("SplitWithSize");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
