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
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Casting.h"

namespace oneflow {
namespace okl {

using namespace user_op;
size_t InferContext::InferTmpSize(user_op::InferContext* ctx) {
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

      auto size = RegContext(reg_op).GetTmpBufferSize();
      max_size = std::max(size, max_size);
    }
  }
  return max_size;
}

InferContext::InferContext(RegContext* reg_ctx) : reg_ctx_(reg_ctx) {}

const TensorDesc* InferContext::LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                                  int32_t index) const {
  return reg_ctx_->TensorDesc4ArgNameAndIndex(arg_name, index);
}

const Shape& InferContext::InputShape(const std::string& arg_name, int32_t index) const {
  return Shape4ArgNameAndIndex(arg_name, index);
}

const Shape& InferContext::Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) const {
  return LogicalTensorDesc4ArgNameAndIndex(arg_name, index)->shape();
}

const std::shared_ptr<const AttrVal>& InferContext::Attr4Name(const std::string& attr_name) const {
  return reg_ctx_->user_op_conf().Attr4Name(attr_name);
}

}  // namespace okl
}  // namespace oneflow