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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/throw.h"
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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace oneflow {

namespace ir {

namespace jit {

static Maybe<mlir::FunctionType> GetFunctionType(user_op::InferContext* ctx,
                                                 mlir::OwningOpRef<mlir::ModuleOp>& module) {
  mlir::func::FuncOp funcOp = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
      module.get(), mlir::SymbolRefAttr::get(module->getContext(), ctx->op_name()));
  CHECK_OR_RETURN(funcOp) << "Fail to find funcOp of symbol " << ctx->op_name();
  const auto funcType = funcOp.getFunctionType();
  CHECK_EQ_OR_RETURN(funcType.getNumInputs(), ctx->input_size("in"))
      << "input size mismatch with mlir assembly";
  CHECK_EQ_OR_RETURN(funcType.getNumResults(), ctx->output_size("out"))
      << "output size mismatch with mlir assembly";
  int32_t arg_i = 0;
  for (mlir::Type arg_type : funcType.getInputs()) {
    if (auto rankedTensorType = arg_type.dyn_cast<mlir::RankedTensorType>()) {
      CHECK_EQ_OR_RETURN(
          (Shape{rankedTensorType.getShape().begin(), rankedTensorType.getShape().end()}),
          ctx->InputShape("in", arg_i))
          << "arg #" << arg_i;
      const auto data_type =
          mlir::oneflow::support::FromMLIRTypeToOFDataType(rankedTensorType.getElementType());
      if (mlir::failed(data_type)) { exit(1); }
      CHECK_EQ_OR_RETURN(data_type.getValue(), ctx->InputDType("in", arg_i)) << "arg #" << arg_i;
      arg_i += 1;
    } else {
      std::string arg_type_str = "";
      llvm::raw_string_ostream os(arg_type_str);
      arg_type.print(os);
      THROW(RuntimeError) << "Unsupported arg type " << arg_type_str;
    }
  }
  return funcType;
}

Maybe<void> SetTensorDataType(user_op::InferContext* ctx) {
  auto mlir_assembly_str = ctx->Attr<std::string>("mlir_assembly");
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::oneflow::OneFlowDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_assembly_str, &context);
  if (!module) {
    LOG(ERROR) << "Fail to load mlir assembly";
    exit(1);
  }

  auto funcType = *JUST(GetFunctionType(ctx, module));
  int32_t res_i = 0;
  for (mlir::Type res_type : funcType.getResults()) {
    if (auto rankedTensorType = res_type.dyn_cast<mlir::RankedTensorType>()) {
      const auto data_type =
          mlir::oneflow::support::FromMLIRTypeToOFDataType(rankedTensorType.getElementType());
      if (mlir::failed(data_type)) { exit(1); }
      ctx->SetDtype4ArgNameAndIndex("out", res_i, data_type.getValue());
      res_i += 1;
    } else {
      std::string res_type_str = "";
      llvm::raw_string_ostream os(res_type_str);
      res_type.print(os);
      THROW(RuntimeError) << "Unsupported arg type " << res_type_str;
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  auto mlir_assembly_str = ctx->Attr<std::string>("mlir_assembly");
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::oneflow::OneFlowDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_assembly_str, &context);
  if (!module) {
    LOG(ERROR) << "Fail to load mlir assembly";
    exit(1);
  }
  auto funcType = *JUST(GetFunctionType(ctx, module));
  int32_t res_i = 0;
  for (mlir::Type res_type : funcType.getResults()) {
    if (auto rankedTensorType = res_type.dyn_cast<mlir::RankedTensorType>()) {
      ctx->SetOutputShape(
          "out", res_i,
          Shape{rankedTensorType.getShape().begin(), rankedTensorType.getShape().end()});

      const auto data_type =
          mlir::oneflow::support::FromMLIRTypeToOFDataType(rankedTensorType.getElementType());
      if (mlir::failed(data_type)) { exit(1); }
      ctx->SetOutputDType("out", res_i, data_type.getValue());
      llvm::SmallVector<int64_t> strides;
      int64_t _;
      auto mem_type =
          mlir::MemRefType::get(rankedTensorType.getShape(), rankedTensorType.getElementType());
      if (failed(mlir::getStridesAndOffset(mem_type, strides, _))) {
        LOG(FATAL) << "Fail to get stride from memory type";
      }
      ctx->SetOutputStride("out", res_i, Stride(strides.begin(), strides.end()));
      res_i += 1;
    } else {
      std::string res_type_str = "";
      llvm::raw_string_ostream os(res_type_str);
      res_type.print(os);
      THROW(RuntimeError) << "Unsupported arg type " << res_type_str;
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace jit

}  // namespace ir

}  // namespace oneflow
