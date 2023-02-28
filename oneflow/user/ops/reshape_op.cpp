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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/user/ops/reshape_user_op_util.h"

namespace oneflow {

/*static*/ Maybe<void> ReshapeOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& in_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();
  const Shape& shape = ctx->Attr<Shape>("shape");
  const auto& outshape = JUST(ReshapeUserOpUtil::GetLogicalOutBlobShape(in_shape, shape));
  user_op::UserOpSbpSignatureBuilder builder = ctx->NewBuilder();
  return ReshapeUserOpUtil::GetReshapeUserOpSbpSignatures(
      in_shape, *outshape, {{"in", 0}}, {{"out", 0}}, ctx->hierarchy_value(), &builder);
}

/*static*/ Maybe<void> ReshapeOp::EnumerateNdSbpSignatures(
    user_op::GetNdSbpSignatureListContext* ctx) {
  const Shape& in_shape = ctx->BlobShape4InputArgNameAndIndex("in", 0);
  const Shape& shape_attr = ctx->Attr<Shape>("shape");
  std::shared_ptr<Shape> out_shape_ptr =
      JUST(ReshapeUserOpUtil::GetLogicalOutBlobShape(in_shape, shape_attr));

  std::vector<NdSbpSignature>* nd_sbp_sig_list = ctx->MutNdSbpSignatureList();
  JUST(ReshapeUserOpUtil::EnumerateNdSbpSignatures({{"in", 0}}, in_shape, {{"out", 0}},
                                                   *out_shape_ptr, ctx->parallel_hierarchy(),
                                                   nd_sbp_sig_list));

  // Go down from the tail to the head, since we might drop the tail.
  for (int32_t sbp_id = nd_sbp_sig_list->size() - 1; sbp_id >= 0; sbp_id--) {
    auto& nd_sbp_sig = (*nd_sbp_sig_list)[sbp_id];
    const auto& out_nd_sbp_it = nd_sbp_sig.bn_in_op2nd_sbp().find("out_0");
    CHECK_OR_RETURN(out_nd_sbp_it != nd_sbp_sig.bn_in_op2nd_sbp().end())
        << "can't get sbp for out_0";
    Shape out_logical_shape = *out_shape_ptr;
    // filter by output only be needed here
    // filter by input will be done in Operator::FilterNdSbpSignatureListByLogicalShape
    if (JUST(FilterNdSbpByLogicalShape(out_nd_sbp_it->second, out_logical_shape,
                                       ctx->parallel_hierarchy()))) {
      // Remove the Nd SBP candidate
      std::swap(nd_sbp_sig, nd_sbp_sig_list->back());
      nd_sbp_sig_list->pop_back();
    }
  }

  DeduplicateNdSbpSignatureList(nd_sbp_sig_list, {"in_0", "out_0"});
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ReshapeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  Shape shape = ctx->Attr<Shape>("shape");
  const user_op::TensorDesc& in_tensor_desc = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor_desc = ctx->MutOutputTensorDesc("out", 0);

  const Shape& in_shape = in_tensor_desc.shape();
  CHECK_OR_RETURN(in_tensor_desc.is_dynamic() == false);  // NOLINT(maybe-need-error-msg)
  out_tensor_desc->set_data_type(in_tensor_desc.data_type());
  if (in_shape.NumAxes() == 0 || shape.NumAxes() == 0) {
    // NOTE(chengcheng): input/output Scalar
    // do nothing
  } else {
    CHECK_GE_OR_RETURN(shape.NumAxes(), 1);     // NOLINT(maybe-need-error-msg)
    CHECK_GE_OR_RETURN(in_shape.NumAxes(), 1);  // NOLINT(maybe-need-error-msg)

    int need_infer_axis = -1;
    size_t count = 1;
    for (int i = 0; i < shape.NumAxes(); ++i) {
      if (shape.At(i) == -1) {
        CHECK_EQ_OR_RETURN(need_infer_axis, -1)
            << Error::RuntimeError() << "Shape " << shape.ToString()
            << " has more than 1 axis that needs to be infered";
        need_infer_axis = i;
      } else {
        count *= shape.At(i);
      }
    }
    if (need_infer_axis != -1) { shape.Set(need_infer_axis, in_shape.elem_cnt() / count); }
  }
  out_tensor_desc->set_shape(shape);
  out_tensor_desc->set_stride(Stride(shape));
  // For 0-size tensor, we don't need to check whether the input and output tensors have the same
  // element size.
  if (in_shape.elem_cnt() > 0) {
    CHECK_EQ_OR_RETURN(shape.elem_cnt(), in_shape.elem_cnt())
        << Error::RuntimeError() << "Reshape infer ERROR! in op_name: " << ctx->op_name()
        << " input shape is : " << in_shape.ToString()
        << " , output shape is : " << shape.ToString()
        << " , and reshape shape conf is : " << ctx->Attr<Shape>("shape").ToString()
        << " op_loc: " << ctx->op_loc();
  }

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ReshapeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  Shape logical_shape = ctx->Attr<Shape>("shape");
  const user_op::TensorDesc& in_tensor_desc = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor_desc = ctx->MutOutputTensorDesc("out", 0);

  const Shape& in_shape = in_tensor_desc.shape();
  out_tensor_desc->set_stride(Stride(in_tensor_desc.shape()));
  out_tensor_desc->set_is_dynamic(in_tensor_desc.is_dynamic());
  if (in_shape.NumAxes() == 0 || logical_shape.NumAxes() == 0) {
    // NOTE(chengcheng): input/output Scalar
    // do nothing
  } else {
    CHECK_GE_OR_RETURN(logical_shape.NumAxes(), 1);  // NOLINT(maybe-need-error-msg)
    CHECK_GE_OR_RETURN(in_shape.NumAxes(), 1);       // NOLINT(maybe-need-error-msg)
    const auto& in_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
    const Shape in_logical_shape =
        *JUST(GetLogicalShape(in_shape, in_nd_sbp, ctx->parallel_desc()));
    int need_infer_axis = -1;
    size_t count = 1;
    for (int i = 0; i < logical_shape.NumAxes(); ++i) {
      if (logical_shape.At(i) == -1) {
        CHECK_EQ_OR_RETURN(need_infer_axis, -1)
            << Error::RuntimeError() << "Shape " << logical_shape.ToString()
            << " has more than 1 axis that needs to be infered";
        need_infer_axis = i;
      } else {
        count *= logical_shape.At(i);
      }
    }
    if (need_infer_axis != -1) {
      logical_shape.Set(need_infer_axis, in_logical_shape.elem_cnt() / count);
    }
  }
  const auto& nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  out_tensor_desc->set_shape(
      *JUST(GetPhysicalShape(logical_shape, nd_sbp, ctx->parallel_desc(), ctx->parallel_ctx())));
  out_tensor_desc->set_stride(Stride(out_tensor_desc->shape()));
  CHECK_EQ_OR_RETURN(out_tensor_desc->shape().elem_cnt(), in_shape.elem_cnt())
      << Error::RuntimeError() << " Reshape infer ERROR! in op_name: " << ctx->op_name()
      << " input shape is : " << in_shape.ToString()
      << " , output shape is : " << out_tensor_desc->shape().ToString()
      << " , output logical shape is " << logical_shape.ToString()
      << " , and reshape shape conf is : " << ctx->Attr<Shape>("shape").ToString()
      << " op_loc: " << ctx->op_loc();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ReshapeOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
