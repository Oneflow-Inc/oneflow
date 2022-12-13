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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/user/kernels/slice_util.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {
bool IsFullSlice(int64_t start, int64_t stop, int64_t step, int64_t size) {
  if (step != 1) { return false; }
  if (start != 0) { return false; }
  if (stop != size) { return false; }
  return true;
}
}  // namespace

/*static*/ Maybe<void> SliceUpdateOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("ref", 0).shape();
  const int64_t ndim = x_shape.NumAxes();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  CHECK_EQ_OR_RETURN(start_vec.size(), ndim)
      << Error::RuntimeError()
      << "The size of start list must be equal to the dimension of ref tensor, "
      << "but got " << start_vec.size() << " and " << ndim;
  CHECK_EQ_OR_RETURN(stop_vec.size(), ndim)
      << Error::RuntimeError()
      << "The size of stop list must be equal to the dimension of ref tensor, "
      << "but got " << stop_vec.size() << " and " << ndim;
  CHECK_EQ_OR_RETURN(step_vec.size(), ndim)
      << Error::RuntimeError()
      << "The size of step list must be equal to the dimension of ref tensor, "
      << "but got " << step_vec.size() << " and " << ndim;

  FOR_RANGE(int64_t, axis, 0, ndim) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("ref", 0), axis)
        .Broadcast(user_op::OpArg("value", 0))
        .Split(user_op::OpArg("y", 0), axis)
        .Build();
    // FullSlice support S+S->S
    if (IsFullSlice(start_vec[axis], stop_vec[axis], step_vec[axis], x_shape.At(axis))) {
      ctx->NewBuilder().Split(ctx->inputs(), axis).Split(ctx->outputs(), axis).Build();
    }
  }
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("ref", 0))
      .PartialSum(user_op::OpArg("value", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceUpdateOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& ref_desc = ctx->InputTensorDesc("ref", 0);
  const Shape& value_shape = ctx->InputTensorDesc("value", 0).shape();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  CHECK_OR_RETURN(!ref_desc.is_dynamic())
      << Error::RuntimeError() << "The ref tensor is not dynamic";
  FOR_RANGE(size_t, i, 0, step_vec.size()) {
    const int64_t step = step_vec.at(i);
    const int64_t start = start_vec.at(i);
    const int64_t stop = stop_vec.at(i);
    CHECK_GT_OR_RETURN(step, 0) << Error::RuntimeError()
                                << "The step list elements must be greater than 0, "
                                << "but got " << step << " at index " << i;

    CHECK_GE_OR_RETURN(start, 0) << Error::RuntimeError()
                                 << "The start list elements must be greater than or equal to 0, "
                                 << "but got " << start << " at index " << i;
    CHECK_GE_OR_RETURN(stop, 0) << Error::RuntimeError()
                                << "The stop list elements must be greater than or equal to 0, "
                                << "but got " << stop << " at index " << i;
    CHECK_LE_OR_RETURN(start, stop) << Error::RuntimeError()
                                    << "The element in start list must be less than or equal to "
                                       "the element in stop list at index "
                                    << i << ", but got " << start << " and " << stop;
    CHECK_EQ_OR_RETURN((stop - start + step - 1) / step, value_shape.At(i))
        << Error::RuntimeError()
        << "The size of slice tuple must be equal to the size of value tensor at dimension " << i
        << ", but got " << (stop - start + step - 1) / step << " and " << value_shape.At(i);
  }
  auto* y_desc = ctx->MutOutputTensorDesc("y", 0);
  y_desc->set_shape(ref_desc.shape());
  y_desc->set_is_dynamic(ref_desc.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceUpdateOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& ref_desc = ctx->InputTensorDesc("ref", 0);
  auto* y_desc = ctx->MutOutputTensorDesc("y", 0);
  y_desc->set_shape(ref_desc.shape());
  y_desc->set_is_dynamic(ref_desc.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceUpdateOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& ref_desc = ctx->InputTensorDesc("ref", 0);
  const user_op::TensorDesc& value_desc = ctx->InputTensorDesc("value", 0);
  CHECK_OR_RETURN(ref_desc.data_type() == value_desc.data_type())
      << Error::TypeError() << "Tensors ref and value must have same type";
  auto* y_desc = ctx->MutOutputTensorDesc("y", 0);
  y_desc->set_data_type(ref_desc.data_type());
  y_desc->set_stride(ref_desc.stride());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SliceOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const Shape& in_shape = input_desc.shape();
  int32_t ndim = in_shape.NumAxes();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  CHECK_EQ_OR_RETURN(start_vec.size(), ndim)
      << "start_vec's dim not equal to ref shape's dim: " << start_vec.size() << " vs " << ndim;
  CHECK_EQ_OR_RETURN(stop_vec.size(), ndim)
      << "stop_vec's dim not equal to ref shape's dim: " << start_vec.size() << " vs " << ndim;
  CHECK_EQ_OR_RETURN(step_vec.size(), ndim)
      << "step_vec's dim not equal to ref shape's dim: " << start_vec.size() << " vs " << ndim;

  FOR_RANGE(int64_t, axis, 0, input_desc.shape().NumAxes()) {
    if (IsFullSlice(start_vec[axis], stop_vec[axis], step_vec[axis], in_shape.At(axis))) {
      ctx->NewBuilder().Split(ctx->inputs(), axis).Split(ctx->outputs(), axis).Build();
    } else {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), axis)
          .PartialSum(user_op::OpArg("y", 0))
          .Build();
    }
  }
  ctx->NewBuilder().PartialSum(user_op::OpArg("x", 0)).PartialSum(user_op::OpArg("y", 0)).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const int64_t ndim = x_shape.NumAxes();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  DimVector dim_vec(ndim);
  FOR_RANGE(size_t, i, 0, dim_vec.size()) {
    const int64_t step = step_vec.at(i);
    const int64_t start = start_vec.at(i);
    const int64_t stop = stop_vec.at(i);
    CHECK_GT_OR_RETURN(step, 0) << Error::RuntimeError()
                                << "The step list elements must be greater than 0, "
                                << "but got " << step << " at index " << i;
    CHECK_GE_OR_RETURN(start, 0) << Error::RuntimeError()
                                 << "The start list elements must be greater than or equal to 0, "
                                 << "but got " << start << " at index " << i;
    CHECK_GE_OR_RETURN(stop, 0) << Error::RuntimeError()
                                << "The stop list elements must be greater than or equal to 0, "
                                << "but got " << stop << " at index " << i;
    CHECK_LE_OR_RETURN(start, stop) << Error::RuntimeError()
                                    << "The element in start list must be less than or equal to "
                                       "the element in stop list at index "
                                    << i << ", but got " << start << " and " << stop;
    const int64_t diff = stop - start - 1;
    dim_vec[i] = diff / step + 1;
  }
  ctx->SetOutputShape("y", 0, Shape(dim_vec));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const int64_t ndim = x_shape.NumAxes();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  DimVector dim_vec(ndim);  // logical shape in slice attributes
  FOR_RANGE(size_t, i, 0, dim_vec.size()) {
    const int64_t step = step_vec[i];
    const int64_t start = start_vec[i];
    const int64_t stop = stop_vec[i];
    CHECK_GT_OR_RETURN(step, 0) << "Slice step must be greater than 0";
    CHECK_GE_OR_RETURN(start, 0) << "Slice start must be greater or equal to 0";
    CHECK_GE_OR_RETURN(stop, 0) << "Slice stop must be greater or equal to 0";
    CHECK_LE_OR_RETURN(start, stop) << "Slice start must be less or equal to stop";
    const int64_t diff = stop - start - 1;
    dim_vec[i] = diff / step + 1;
  }
  // Get physical shape with TensorSliceView
  const NdSbp& y_nd_sbp = ctx->NdSbp4ArgNameAndIndex("y", 0);
  const Shape& parallel_hierarchy = *ctx->parallel_desc().hierarchy();
  const Shape& logical_shape = Shape(dim_vec);
  const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
  const TensorSliceView& slice_view =
      GetTensorSliceView4ParallelId(parallel_hierarchy, y_nd_sbp, logical_shape, parallel_id);
  ctx->SetOutputShape("y", 0, Shape(slice_view.shape()));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SliceGradOp::GetSbp(user_op::SbpContext* ctx) {
  const Shape& like_shape = ctx->Attr<Shape>("like_shape");
  const int64_t ndim = like_shape.NumAxes();
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  CHECK_EQ_OR_RETURN(start_vec.size(), ndim)
      << Error::RuntimeError()
      << "The size of start list must be equal to the dimension of ref tensor, "
      << "but got " << start_vec.size() << " and " << ndim;
  CHECK_EQ_OR_RETURN(stop_vec.size(), ndim)
      << Error::RuntimeError()
      << "The size of stop list must be equal to the dimension of ref tensor, "
      << "but got " << stop_vec.size() << " and " << ndim;
  CHECK_EQ_OR_RETURN(step_vec.size(), ndim)
      << Error::RuntimeError()
      << "The size of step list must be equal to the dimension of ref tensor, "
      << "but got " << step_vec.size() << " and " << ndim;
  FOR_RANGE(int, i, 0, ndim) {
    if (IsFullSlice(start_vec[i], stop_vec[i], step_vec[i], like_shape.At(i))) {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
    }
  }
  ctx->NewBuilder().PartialSum(user_op::OpArg("dy", 0)).PartialSum(user_op::OpArg("dx", 0)).Build();
  ctx->NewBuilder().Broadcast(user_op::OpArg("dy", 0)).Broadcast(user_op::OpArg("dx", 0)).Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& like_shape = ctx->Attr<Shape>("like_shape");
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");

  const int64_t ndim = dy_shape.NumAxes();
  CHECK_EQ_OR_RETURN(start_vec.size(), ndim)
      << Error::RuntimeError()
      << "The size of start list must be equal to the dimension of ref tensor, "
      << "but got " << start_vec.size() << " and " << ndim;
  CHECK_EQ_OR_RETURN(stop_vec.size(), ndim)
      << Error::RuntimeError()
      << "The size of stop list must be equal to the dimension of ref tensor, "
      << "but got " << stop_vec.size() << " and " << ndim;
  CHECK_EQ_OR_RETURN(step_vec.size(), ndim)
      << Error::RuntimeError()
      << "The size of step list must be equal to the dimension of ref tensor, "
      << "but got " << step_vec.size() << " and " << ndim;
  ctx->SetOutputShape("dx", 0, like_shape);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  Shape logical_shape = ctx->Attr<Shape>("like_shape");
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  user_op::TensorDesc* dx_desc = ctx->MutOutputTensorDesc("dx", 0);
  dx_desc->set_is_dynamic(dy_desc.is_dynamic());

  const auto& nd_sbp = ctx->NdSbp4ArgNameAndIndex("dx", 0);
  dx_desc->set_shape(
      *JUST(GetPhysicalShape(logical_shape, nd_sbp, ctx->parallel_desc(), ctx->parallel_ctx())));
  int dx_ndim = dx_desc->shape().NumAxes();
  int dy_ndim = dy_desc.shape().NumAxes();
  CHECK_EQ_OR_RETURN(dx_ndim, dy_ndim)
      << Error::RuntimeError() << "The output dimension (" << dx_ndim
      << ") should be equal to the input dimension (" << dy_ndim << ") for slice backward";
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceGradOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("dx", 0, ctx->InputDType("dy", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SliceGradOp::ModifyInputArg(const GetInputArgModifier& GetInputArgModifierFn,
                                                   const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* dy_modifier = GetInputArgModifierFn("dy", 0);
  dy_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
