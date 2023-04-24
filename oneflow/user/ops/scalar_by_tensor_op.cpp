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

namespace oneflow {

namespace {

Maybe<void> TensorDescInferFn(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& scalar = ctx->InputTensorDesc("scalar", 0);
  CHECK_EQ_OR_RETURN(scalar.shape().elem_cnt(), 1)
      << Error::RuntimeError() << "The input scalar tensor is not a scalar";
  user_op::TensorDesc* y = ctx->MutOutputTensorDesc("y", 0);
  y->set_shape(x.shape());
  y->set_is_dynamic(x.is_dynamic());
  return Maybe<void>::Ok();
}

Maybe<void> DataTypeInferFn(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& scalar = ctx->InputTensorDesc("scalar", 0);
  CHECK_EQ_OR_RETURN(x.data_type(), scalar.data_type())
      << Error::TypeError() << "Tensors x and scalar have different type";
  user_op::TensorDesc* y = ctx->MutOutputTensorDesc("y", 0);
  y->set_data_type(x.data_type());
  return Maybe<void>::Ok();
}

Maybe<void> GetBasicSbpSignature(user_op::SbpContext* ctx) {
  const auto& x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Broadcast(user_op::OpArg("scalar", 0))
        .Build();
  }
  return Maybe<void>::Ok();
}

using GetSbpFn = std::function<Maybe<void>(user_op::SbpContext*)>;
GetSbpFn MakeGetSbpFn(GetSbpFn extra) {
  return [extra](user_op::SbpContext* ctx) -> Maybe<void> {
    JUST(extra(ctx));
    JUST(GetBasicSbpSignature(ctx));
    return Maybe<void>::Ok();
  };
}

}  // namespace

/*static*/ Maybe<void> ScalarAddByTensorOp::GetSbp(user_op::SbpContext* ctx) {
  return MakeGetSbpFn([](user_op::SbpContext* ctx) {
    ctx->NewBuilder()
        .PartialSum(user_op::OpArg("x", 0))
        .PartialSum(user_op::OpArg("scalar", 0))
        .PartialSum(user_op::OpArg("y", 0))
        .Build();
    return Maybe<void>::Ok();
  })(ctx);
}
/*static*/ Maybe<void> ScalarAddByTensorOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return TensorDescInferFn(ctx);
}
/*static*/ Maybe<void> ScalarAddByTensorOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ScalarAddByTensorOp::InferDataType(user_op::InferContext* ctx) {
  return DataTypeInferFn(ctx);
}

/*static*/ Maybe<void> HostScalarAddByTensorOp::GetSbp(user_op::SbpContext* ctx) {
  return MakeGetSbpFn([](user_op::SbpContext* ctx) {
    ctx->NewBuilder()
        .PartialSum(user_op::OpArg("x", 0))
        .PartialSum(user_op::OpArg("scalar", 0))
        .PartialSum(user_op::OpArg("y", 0))
        .Build();
    return Maybe<void>::Ok();
  })(ctx);
}
/*static*/ Maybe<void> HostScalarAddByTensorOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return TensorDescInferFn(ctx);
}
/*static*/ Maybe<void> HostScalarAddByTensorOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> HostScalarAddByTensorOp::InferDataType(user_op::InferContext* ctx) {
  return DataTypeInferFn(ctx);
}

REGISTER_OP_HOST_MEMORY_INPUT("host_scalar_add_by_tensor", "scalar", 0);

/*static*/ Maybe<void> ScalarSubByTensorOp::GetSbp(user_op::SbpContext* ctx) {
  return MakeGetSbpFn([](user_op::SbpContext* ctx) {
    ctx->NewBuilder()
        .PartialSum(user_op::OpArg("x", 0))
        .PartialSum(user_op::OpArg("scalar", 0))
        .PartialSum(user_op::OpArg("y", 0))
        .Build();
    return Maybe<void>::Ok();
  })(ctx);
}
/*static*/ Maybe<void> ScalarSubByTensorOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return TensorDescInferFn(ctx);
}
/*static*/ Maybe<void> ScalarSubByTensorOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ScalarSubByTensorOp::InferDataType(user_op::InferContext* ctx) {
  return DataTypeInferFn(ctx);
}

/*static*/ Maybe<void> ScalarMulByTensorOp::GetSbp(user_op::SbpContext* ctx) {
  return MakeGetSbpFn([](user_op::SbpContext* ctx) {
    ctx->NewBuilder()
        .PartialSum(user_op::OpArg("x", 0))
        .Broadcast(user_op::OpArg("scalar", 0))
        .PartialSum(user_op::OpArg("y", 0))
        .Build();
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("x", 0))
        .PartialSum(user_op::OpArg("scalar", 0))
        .PartialSum(user_op::OpArg("y", 0))
        .Build();
    return Maybe<void>::Ok();
  })(ctx);
}
/*static*/ Maybe<void> ScalarMulByTensorOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return TensorDescInferFn(ctx);
}
/*static*/ Maybe<void> ScalarMulByTensorOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ScalarMulByTensorOp::InferDataType(user_op::InferContext* ctx) {
  return DataTypeInferFn(ctx);
}

/*static*/ Maybe<void> ScalarDivByTensorOp::GetSbp(user_op::SbpContext* ctx) {
  return MakeGetSbpFn([](user_op::SbpContext* ctx) {
    ctx->NewBuilder()
        .PartialSum(user_op::OpArg("x", 0))
        .Broadcast(user_op::OpArg("scalar", 0))
        .PartialSum(user_op::OpArg("y", 0))
        .Build();
    return Maybe<void>::Ok();
  })(ctx);
}
/*static*/ Maybe<void> ScalarDivByTensorOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return TensorDescInferFn(ctx);
}
/*static*/ Maybe<void> ScalarDivByTensorOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ScalarDivByTensorOp::InferDataType(user_op::InferContext* ctx) {
  return DataTypeInferFn(ctx);
}

}  // namespace oneflow
