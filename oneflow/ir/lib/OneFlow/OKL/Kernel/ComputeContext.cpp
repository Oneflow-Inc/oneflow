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
#include "OneFlow/OKL/Kernel/ComputeContext.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "OneFlow/OKL/OKLOps.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {
namespace okl {

user_op::Tensor* ComputeContext::CreateTensorWithArgNameAndIndex(const std::string& arg_name,
                                                                 int32_t index) {
  auto op = reg_ctx_->GetOp();
  auto source = mlir::oneflow::user_op::GetOpSourceByName(op, arg_name);

  if (source.type == mlir::oneflow::user_op::Source::OUTPUT) {
    if (op->getNumResults() <= index + source.offset) { return nullptr; }
    mlir::Value val = op->getResult(index + source.offset);
    auto use = *val.getUsers().begin();
    if (auto ret_op = llvm::dyn_cast_or_null<mlir::okl::GetTensorAsRetOp>(use)) {
      return comp_ctx_->Tensor4ArgNameAndIndex("out", ret_op.index());
    }
    if (auto pool_op = llvm::dyn_cast_or_null<mlir::okl::TensorToPoolOp>(use)) {
      return tmp_buffer_.GetPoolTensor(TensorDesc4ArgNameAndIndex(arg_name, index),
                                       pool_op.offset());
    }
    op->emitError("Failed to find " + std::to_string(index) + "in outputs");
    exit(1);
  }

  if (source.type == mlir::oneflow::user_op::Source::INPUT) {
    if (op->getNumOperands() <= index + source.offset) { return nullptr; }
    mlir::Value val = op->getOperand(index + source.offset);
    auto define_op = val.getDefiningOp();
    return llvm::TypeSwitch<::mlir::Operation*, user_op::Tensor*>(define_op)
        .Case([&](mlir::okl::GetTensorFromArgOp elem) {
          return comp_ctx_->Tensor4ArgNameAndIndex("in", elem.index());
        })
        .Case([&](mlir::okl::GetTensorFromRetOp elem) {
          return comp_ctx_->Tensor4ArgNameAndIndex("out", elem.index());
        })
        .Case([&](mlir::okl::PoolToTensorOp elem) {
          return tmp_buffer_.GetPoolTensor(TensorDesc4ArgNameAndIndex(arg_name, index),
                                           elem.offset());
        })
        .Default([&](::mlir::Operation* op) {
          op->dump();
          LOG(FATAL) << "Signature: " << arg_name << " Not supported";
          return nullptr;
        });
  }

  if (source.type == mlir::oneflow::user_op::Source::BUFFER) {
    auto wrap = op->getParentOfType<mlir::okl::WrapperKernelOp>();
    for (auto& op : wrap.body().front()) {
      if (auto pool_to_buffer = llvm::dyn_cast_or_null<mlir::okl::PoolToBufferOp>(op)) {
        return tmp_buffer_.GetPoolBuffer(pool_to_buffer.getType().getShape()[0],
                                         pool_to_buffer.offset());
      }
    }
  }

  op->emitError("Failed to check source type");
  exit(1);
}
user_op::Tensor* ComputeContext::Tensor4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) {
  auto it = tensor_.find({arg_name, index});
  if (it != tensor_.end()) return it->second;
  user_op::Tensor* res = CreateTensorWithArgNameAndIndex(arg_name, index);
  tensor_[{arg_name, index}] = res;
  return res;
}

}  // namespace okl

}  // namespace oneflow