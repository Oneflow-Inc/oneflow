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
#include "llvm/Support/raw_ostream.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/user/ops/nn_util.h"
#include "OneFlow/OKL/Kernel/JITOpInfer.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

namespace oneflow {

namespace {

Maybe<void> GetSbpFn(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> MlirJitOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return ir::jit::InferTensorDesc(ctx);
}

Maybe<void> MlirJitOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return ir::jit::InferTensorDesc(ctx);
}

Maybe<void> MlirJitOp::GetSbp(user_op::SbpContext* ctx) { return GetSbpFn(ctx); }

Maybe<void> MlirJitOp::InferDataType(user_op::InferContext* ctx) {
  return ir::jit::SetTensorDataType(ctx);
  ;
}

}  // namespace oneflow
