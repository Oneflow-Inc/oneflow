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

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowSupport.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/user/ops/nn_util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  auto mlir_assembly_str = ctx->Attr<std::string>("mlir_assembly");
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::oneflow::OneFlowDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_assembly_str, &context);

  auto args_it = module->getBodyRegion().getOps().begin()->getRegions().begin()->args_begin();
  auto in0 = args_it->getType().dyn_cast<mlir::RankedTensorType>();
  auto in1 = (++args_it)->getType().dyn_cast<mlir::RankedTensorType>();

  CHECK_EQ(Shape(in0.getShape().begin(), in0.getShape().end()), ctx->InputShape("in", 0));
  CHECK_EQ(mlir::oneflow::support::GetDataTypeFromMLIRType(in1.getElementType()),
           ctx->InputDType("in", 1));

  CHECK_EQ(ctx->inputs().size(), 2);
  CHECK_EQ(ctx->outputs().size(), 1);
  const Shape& in_shape = ctx->InputShape("in", 0);
  Shape* out_shape = ctx->MutOutputShape("out", 0);
  *out_shape = in_shape;
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 1);
  return Maybe<void>::Ok();
}

Maybe<void> GetSbpFn(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

Maybe<void> InferDataTypeFn(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> MlirJitOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc(ctx);
}

Maybe<void> MlirJitOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc(ctx);
}

Maybe<void> MlirJitOp::GetSbp(user_op::SbpContext* ctx) { return GetSbpFn(ctx); }

Maybe<void> MlirJitOp::InferDataType(user_op::InferContext* ctx) { return InferDataTypeFn(ctx); }

}  // namespace oneflow
