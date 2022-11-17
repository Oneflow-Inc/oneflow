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
#include "oneflow/ir/oneflow-extension/include/OneFlow/kernel_launch/InferMisc/InferContext.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/kernel_launch/TmpBufferManager.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Casting.h"

namespace oneflow {
namespace okl {

using namespace user_op;
std::unordered_map<std::string, size_t> TmpBufferManager::off_set_list;
std::shared_ptr<TmpBufferManager> TmpBufferManager::InitTmpBufferManager(user_op::Tensor* tensor) {
  return std::make_shared<TmpBufferManager>(tensor);
}

size_t TmpBufferManager::InferTmpSize(user_op::InferContext* ctx) {
  mlir::MLIRContext mlir_ctx(KernelLaunchState::GetRegistry());

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(ctx->Attr<std::string>("mlir_assembly"), &mlir_ctx);
  if (!module) { LOG(FATAL) << "Fail to load mlir assembly"; }
  if (failed(mlir::okl::LowerWrapOpsToOKL(*module))) {
    LOG(ERROR) << "Fail lowering kernel launch Module to okl ir";
    exit(1);
  }

  auto& ops = module->lookupSymbol("okl_init_context")->getRegion(0).front();

  size_t max_size = 0;
  for (auto& op : ops) {
    if (llvm::dyn_cast_or_null<mlir::okl::BuildRegContextOp>(op)) {
      mlir::Operation* reg_op = nullptr;
      for (auto& op_it : op.getRegion(0).front().getOperations()) {
        if (op_it.getDialect()->getNamespace() == "oneflow") {
          reg_op = &op_it;
          break;
        }
      }
      if (!reg_op) { LOG(FATAL) << "Failed to find reg_op in okl.build_reg_context_op"; }

      auto op_name = reg_op->getAttr("op_name").dyn_cast<mlir::StringAttr>().str();
      off_set_list[op_name] = max_size;
      auto size = RegContext(reg_op).GetTmpBufferSize();
      max_size += size;
    }
  }
  return max_size;
}

TmpBufferManager::TmpBufferManager(user_op::Tensor* tensor) {
  for (const auto& it : off_set_list) {
    manager_.emplace(it.first, TmpBufferMapper(tensor, it.second));
  }
}

user_op::Tensor* TmpBufferManager::FetchTmpBuffer(std::string& op_name) {
  if (auto val = manager_.find(op_name); val != manager_.end()) { return &val->second; }
  LOG(FATAL) << op_name << " not found in tmp buffer manager.";
}

}  // namespace okl
}  // namespace oneflow
