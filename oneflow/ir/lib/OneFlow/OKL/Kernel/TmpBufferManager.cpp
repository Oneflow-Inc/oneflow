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
#include "OneFlow/OKL/Kernel/TmpBufferManager.h"
#include "OneFlow/OKL/Kernel/LauncherState.h"
#include "OneFlow/OKL/OKLOps.h"
#include "OneFlow/OKM/Conversion/Conversion.h"
#include "OneFlow/OKM/passes.h"
#include "OneFlow/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Casting.h"

namespace oneflow {
namespace okl {

size_t TmpBufferManager::InferTmpSize(user_op::InferContext* ctx) {
  using namespace user_op;
  mlir::MLIRContext mlir_ctx(GetRegistry());

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(ctx->Attr<std::string>("mlir_assembly"), &mlir_ctx);
  if (!module) { LOG(FATAL) << "Fail to load mlir assembly"; }
  if (failed(mlir::okm::LowerWrapOpsToOKL(*module))) {
    LOG(ERROR) << "Fail lowering kernel launch Module to okl ir";
    exit(1);
  }

  size_t pool_size = 0;
  module->walk([&](mlir::func::FuncOp op) {
    if (op.getSymName().startswith(mlir::okm::func_name::OKL_GRAPH_NAME)) {
      if (auto pool_size_attr =
              op->getAttrOfType<mlir::IntegerAttr>(mlir::okm::func_name::OKL_POOL_SIZE_TAG)) {
        pool_size = pool_size_attr.getInt();
      }
    }
  });
  return pool_size;
}

}  // namespace okl
}  // namespace oneflow
