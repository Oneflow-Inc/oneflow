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

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class ConstantFunctor {
 public:
  ConstantFunctor() { op_ = CHECK_JUST(one::OpBuilder("constant").Output("out").Build()); }
  Maybe<Tensor> operator()(const Shape& shape, const Scalar& value, const DataType& dtype) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", shape));
    JUST(attrs.SetAttr<DataType>("dtype", dtype));
    if (IsIntegralDataType(dtype)) {
      JUST(attrs.SetAttr<bool>("is_floating_value", false));
      JUST(attrs.SetAttr<int64_t>("integer_value", JUST(value.As<int64_t>())));
    } else {
      JUST(attrs.SetAttr<bool>("is_floating_value", true));
      JUST(attrs.SetAttr<double>("floating_value", JUST(value.As<double>())));
    }
    return OpInterpUtil::Dispatch<Tensor>(*op_, {}, attrs);
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

class ArgWhereFunctor {
 public:
  ArgWhereFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("argwhere").Input("input").Output("output").Output("output_size").Build());
  }
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& x,
                                const DataType& dtype) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<DataType>("dtype", dtype));
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

class ExpandFunctor {
 public:
  ExpandFunctor() { op_ = CHECK_JUST(one::OpBuilder("expand").Input("in").Output("out").Build()); }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<int32_t>& in_shape,
                           const std::vector<int32_t>& out_shape,
                           const std::vector<int32_t>& stride) const {
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

class SliceFunctor : public SliceBaseFunctor {
 public:
  SliceFunctor() { op_ = CHECK_JUST(one::OpBuilder("slice").Input("x").Output("y").Build()); }
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

class UnsortedBatchSegmentSumFunctor {
 public:
  UnsortedBatchSegmentSumFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("unsorted_batch_segment_sum").Input("data").Input("segment_ids").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, 
                           const std::shared_ptr<one::Tensor>& segment_ids, 
                           const int32_t& num_segments) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("num_segments", num_segments));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {x, segment_ids}, attrs);
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
    const auto& regular_index = JUST(RegularTensorIndex(index, *(x->shape())));
    int64_t ndims = x->shape()->NumAxes();
    CHECK_GE_OR_RETURN(regular_index->size(), ndims) << "Tensor index failed to be regularlized.";
    std::vector<int64_t> start(ndims), end(ndims), step(ndims);
    int dim = 0;
    DimVector result_dims;
    for (int i = 0; i < regular_index->size(); ++i) {
      const auto& index_item = regular_index->at(i);
      CHECK_OR_RETURN(!index_item.IsEllipsis())
          << "Tensor index should not have ellipsis once regularlized.";
      if (index_item.IsSlice()) {
        CHECK_LT_OR_RETURN(dim, ndims);
        start[dim] = index_item.slice().start();
        end[dim] = index_item.slice().end();
        step[dim] = index_item.slice().step();
        int64_t length = (end[dim] - start[dim] + step[dim] - 1) / step[dim];
        result_dims.emplace_back(length);
        dim++;
      } else if (index_item.IsInteger()) {
        CHECK_LT_OR_RETURN(dim, ndims);
        start[dim] = index_item.integer();
        end[dim] = start[dim] + 1;
        step[dim] = 1;
        dim++;
      } else if (index_item.IsNone()) {
        result_dims.emplace_back(1);
      } else if (index_item.IsBoolean()) {
        CHECK_OR_RETURN(index_item.boolean()) << "Index false is not supported.";
        result_dims.emplace_back(1);
      }
    }
    CHECK_EQ_OR_RETURN(dim, ndims)
        << "Specified dims count for regularlized tensor index should equal to tensor dimension "
        << ndims;

    bool is_identity = [&]() {
      for (int i = 0; i < ndims; ++i) {
        if (start[i] != 0 || end[i] != x->shape()->At(i) || step[i] != 1) { return false; }
      }
      return true;
    }();
    std::shared_ptr<one::Tensor> result;
    if (is_identity) {
      result = JUST(functional::Copy(x, JUST(x->device())->type(), JUST(x->device())->device_id()));
    } else {
      result = JUST(functional::Slice(x, start, end, step));
    }

    Shape shape(result_dims);
    if (shape.NumAxes() != 0 && shape != *(result->shape())) {
      return functional::Reshape(result, shape);
    }
    return result;
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ConstantFunctor>("Constant");
  m.add_functor<impl::ZerosLikeFunctor>("ZerosLike");
  m.add_functor<impl::OnesLikeFunctor>("OnesLike");
  m.add_functor<impl::FlattenFunctor>("Flatten");
  m.add_functor<impl::WhereFunctor>("Where");
  m.add_functor<impl::ArgWhereFunctor>("ArgWhere");
  m.add_functor<impl::BroadcastLikeFunctor>("BroadcastLike");
  m.add_functor<impl::ConcatFunctor>("Concat");
  m.add_functor<impl::ExpandFunctor>("Expand");
  m.add_functor<impl::ExpandDimsFunctor>("ExpandDims");
  m.add_functor<impl::GatherFunctor>("Gather");
  m.add_functor<impl::DimGatherFunctor>("DimGather");
  m.add_functor<impl::GatherNdFunctor>("GatherNd");
  m.add_functor<impl::ScatterNdLikeFunctor>("ScatterNdLike");
  m.add_functor<impl::ReshapeFunctor>("Reshape");
  m.add_functor<impl::SliceFunctor>("Slice");
  m.add_functor<impl::LogicalSliceAssignFunctor>("LogicalSliceAssign");
  m.add_functor<impl::LogicalSliceFunctor>("LogicalSlice");
  m.add_functor<impl::SliceUpdateFunctor>("SliceUpdate");
  m.add_functor<impl::SqueezeFunctor>("Squeeze");
  m.add_functor<impl::CopyFunctor>("Copy");
  m.add_functor<impl::UpsampleFunctor>("Upsample");
  m.add_functor<impl::UnsortedSegmentSumLikeFunctor>("UnsortedSegmentSumLike");
  m.add_functor<impl::UnsortedBatchSegmentSumFunctor>("UnsortedBatchSegmentSum");
  m.add_functor<impl::TriuFunctor>("Triu");
  m.add_functor<impl::DiagFunctor>("Diag");
  m.add_functor<impl::DiagGradFunctor>("DiagGrad");
  m.add_functor<impl::TensorGetItemFunctor>("TensorGetItem");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
