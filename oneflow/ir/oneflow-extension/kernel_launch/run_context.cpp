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
#include "OneFlow/UserOpReflection.h"
#include "OneFlow/OKL/OKLOps.h"
#include "OneFlow/kernel_launch/RegContext.h"
#include "OneFlow/kernel_launch/RunContext.h"
#include "OneFlow/kernel_launch/InferMisc/InitContext.h"
#include "OneFlow/kernel_launch/InferMisc/InferContext.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>

namespace oneflow {
namespace okl {

RunContext::RunContext(std::shared_ptr<RegContext> reg, user_op::KernelComputeContext* comp)
    : reg_ctx_(std::move(reg)), comp_ctx_(comp) {
  InitContext init_ctx(this);
  kernel_state_ = reg_ctx_->kernel_->CreateOpKernelState(&init_ctx);
  kernel_cache_ = reg_ctx_->kernel_->InitOpKernelCache(&init_ctx);

  tmp_buffer_manager_ =
      TmpBufferManager::InitTmpBufferManager(comp_ctx_->Tensor4ArgNameAndIndex("tmp_buffer", 0));
}

const user_op::TensorDesc* RunContext::TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                                  int32_t index) const {
  return reg_ctx_->TensorDesc4ArgNameAndIndex(arg_name, index);
}

user_op::Tensor* RunContext::Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) {
  auto op = reg_ctx_->GetOp();
  using namespace mlir::oneflow::user_op;
  auto source = GetOpSourceByName(op, arg_name);

  if (source.type == Source::OUTPUT) {
    if (op->getNumResults() <= index + source.offset) { return nullptr; }
    mlir::Value val = op->getResult(index + source.offset);
    for (auto use : val.getUsers()) {
      if (llvm::isa<mlir::okl::GetTensorAsRetOp>(use)) {
        auto index = use->getAttr("index").cast<mlir::IntegerAttr>().getInt();
        return comp_ctx_->Tensor4ArgNameAndIndex("out", index);
      }
    }
    op->emitError("Failed to find " + std::to_string(index) + "in outputs");
    exit(1);
  }

  if (source.type == Source::INPUT) {
    if (op->getNumOperands() <= index + source.offset) { return nullptr; }
    mlir::Value val = op->getOperand(index + source.offset);
    auto define_op = val.getDefiningOp();
    return llvm::TypeSwitch<::mlir::Operation*, user_op::Tensor*>(define_op)
        .Case([&](mlir::okl::GetTensorFromArgOp elem) {
          auto index = elem.index();
          return comp_ctx_->Tensor4ArgNameAndIndex("in", index);
        })
        .Case([&](mlir::okl::GetTensorFromRetOp elem) {
          auto index = elem.index();
          return comp_ctx_->Tensor4ArgNameAndIndex("out", index);
        })
        .Default([&](::mlir::Operation* op) {
          LOG(FATAL) << "Signature: " << arg_name << " Not supported";
          return nullptr;
        });
  }

  if (source.type == Source::BUFFER) {
    auto op_name = op->getAttr("op_name").dyn_cast<mlir::StringAttr>().str();
    return tmp_buffer_manager_->GetBufferTensor();
  }

  op->emitError("Failed to check source type");
  exit(1);
}

ep::Stream* RunContext::stream() { return comp_ctx_->stream(); }

DeviceType RunContext::device_type() const { return reg_ctx_->device_type(); }
const ParallelContext& RunContext::parallel_ctx() const { return comp_ctx_->parallel_ctx(); }

const ArgVec& RunContext::inputs() const { return reg_ctx_->inputs(); }
const ArgVec& RunContext::outputs() const { return reg_ctx_->outputs(); }

const user_op::UserOpConfWrapper& RunContext::user_op_conf() const {
  return reg_ctx_->user_op_conf();
}

const std::shared_ptr<const user_op::AttrVal>& RunContext::Attr4Name(
    const std::string& attr_name) const {
  return user_op_conf().Attr4Name(attr_name);
}

}  // namespace okl
}  // namespace oneflow
