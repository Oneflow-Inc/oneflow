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
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

Maybe<void> InferTensorDescFn(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input_tensor", 0);
  const auto& reduce_axes = ctx->Attr<std::vector<int32_t>>("axis");
  Shape output_shape;
  // For 0-dim Tensor
  if (reduce_axes.empty()) {
    output_shape = input_shape;
  } else {
    const AxisVector reduce_axes_vec = {reduce_axes.begin(), reduce_axes.end()};
    const Shape& reduce_shape = CreateReducedShape(input_shape, reduce_axes_vec);
    const bool keepdims = ctx->Attr<bool>("keepdims");
    if (keepdims) {
      output_shape = reduce_shape;
    } else {
      output_shape = reduce_shape.RemoveOnes(reduce_axes_vec);
    }
  }
  ctx->SetOutputShape("output_tensor", 0, output_shape);
  ctx->SetOutputStride("output_tensor", 0, Stride(output_shape));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("output_tensor", 0, ctx->InputDType("input_tensor", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferLogicalDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("output_tensor", 0, DataType::kBool);
  return Maybe<void>::Ok();
}

template<template<typename> class binary_func>
void GeneratePartialSbp(user_op::SbpContext* ctx, int64_t axis) {
  // TODO(lixinqi)
}

template<>
void GeneratePartialSbp<BinaryFuncSum>(user_op::SbpContext* ctx, int64_t axis) {
  ctx->NewBuilder().Split(ctx->inputs(), axis).PartialSum(ctx->outputs()).Build();
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
}

template<template<typename> class binary_func>
Maybe<void> GetSbpFn(user_op::SbpContext* ctx) {
  const auto& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("input_tensor", 0);
  int64_t num_axes = in_tensor.shape().NumAxes();
  bool keep_dims = ctx->Attr<bool>("keepdims");
  const auto& reduce_axes = ctx->Attr<std::vector<int32_t>>("axis");
  HashSet<int32_t> conf_axes;
  ReduceSbpUtil::GetRegularAxes(num_axes, reduce_axes, &conf_axes);
  auto IsReducedAxis = ReduceSbpUtil::MakePredicatorIsReducedAxis(conf_axes, num_axes);
  int32_t num_reduced_axes = 0;
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i)) {
      GeneratePartialSbp<binary_func>(ctx, i);
      num_reduced_axes += 1;
    } else {
      ctx->NewBuilder()
          .Split(ctx->inputs(), i)
          .Split(ctx->outputs(), keep_dims ? i : i - num_reduced_axes)
          .Build();
    }
  }
  if (num_axes == 0) {
    ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  }
  return Maybe<void>::Ok();
}

#define IMPLEMENT_REDUCE_OP_FUNCS(name, binary_func, infer_dtype_func)                   \
  /*static*/ Maybe<void> name##Op::GetSbp(user_op::SbpContext* ctx) {                    \
    return GetSbpFn<binary_func>(ctx);                                                   \
  }                                                                                      \
  /*static*/ Maybe<void> name##Op::InferLogicalTensorDesc(user_op::InferContext* ctx) {  \
    return InferTensorDescFn(ctx);                                                       \
  }                                                                                      \
  /*static*/ Maybe<void> name##Op::InferPhysicalTensorDesc(user_op::InferContext* ctx) { \
    return InferLogicalTensorDesc(ctx);                                                  \
  }                                                                                      \
  /*static*/ Maybe<void> name##Op::InferDataType(user_op::InferContext* ctx) {           \
    return infer_dtype_func(ctx);                                                        \
  }

IMPLEMENT_REDUCE_OP_FUNCS(ReduceAny, BinaryFuncAny, InferLogicalDataType)
IMPLEMENT_REDUCE_OP_FUNCS(ReduceAll, BinaryFuncAll, InferLogicalDataType)
IMPLEMENT_REDUCE_OP_FUNCS(ReduceMin, BinaryFuncMin, oneflow::InferDataType)
IMPLEMENT_REDUCE_OP_FUNCS(ReduceMax, BinaryFuncMax, oneflow::InferDataType)
IMPLEMENT_REDUCE_OP_FUNCS(ReduceSum, BinaryFuncSum, oneflow::InferDataType)
IMPLEMENT_REDUCE_OP_FUNCS(ReduceProd, BinaryFuncProd, oneflow::InferDataType)
IMPLEMENT_REDUCE_OP_FUNCS(ReduceNanSum, BinaryFuncNanSum, oneflow::InferDataType)
#undef IMPLEMENT_REDUCE_OP_FUNCS

}  // namespace oneflow
