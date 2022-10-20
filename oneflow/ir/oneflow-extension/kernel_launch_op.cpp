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

#include "OneFlow/OKL/OKLDialect.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/UserOpReflection.h"
#include "OneFlow/Passes.h"
#include "OneFlow/Extension.h"
#include "OneFlow/OKL/Conversion/Conversion.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"
#include "oneflow/core/kernel/blob_tensor_view.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/JITOpInfer.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/JITEngine.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

#include <cmath>
#include <memory>
#include <tuple>
#include <utility>
#include <sys/types.h>

namespace oneflow {

Maybe<void> KernelLaunchOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return ir::jit::InferTensorDesc(ctx);
}

Maybe<void> KernelLaunchOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return ir::jit::InferTensorDesc(ctx);
}

Maybe<void> KernelLaunchOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

Maybe<void> KernelLaunchOp::InferDataType(user_op::InferContext* ctx) {
  return ir::jit::SetTensorDateType(ctx);
}

namespace {

// this context should support querying information about the kernel from representation in MLIR
using ArgVec = std::vector<std::pair<std::string, int32_t>>;
class KernelLaunchOpKernelRegContext final : public user_op::KernelRegContext {
 public:
  explicit KernelLaunchOpKernelRegContext(::mlir::ModuleOp module_op) : owned_module_(module_op) {
    module_op.getBody()->walk([&](::mlir::func::FuncOp func_op) {
      func_op_ = func_op;
      return ::mlir::WalkResult::interrupt();
    });
    if (!func_op_) { LOG(FATAL) << "FuncOp not found in module"; }

    // init blob descriptors
    auto& body = func_op_->getRegion(0);
    auto& block = body.front();
    CHECK(!block.empty());
    auto& op = block.front();
    op_ = &op;
    for (const auto& operand_id : ::llvm::enumerate(
             ::mlir::oneflow::user_op::ArgIds<mlir::OpTrait::AttrSizedOperandSegments>(&op))) {
      user_op::NaiveTensorDesc tensor_desc{};
      auto operand = op.getOperand(operand_id.index());
      if (auto rankedTensorType = operand.getType().dyn_cast<mlir::RankedTensorType>()) {
        tensor_desc.set_shape(
            Shape{rankedTensorType.getShape().begin(), rankedTensorType.getShape().end()});
        tensor_desc.set_data_type(
            mlir::oneflow::support::GetDataTypeFromMLIRType(rankedTensorType.getElementType()));
        // TODO: set stride
        // TODO: set is_dynamic
      } else {
        LOG(FATAL) << "Unranked tensor type not supported";
      }
      CHECK(arg2tensor_desc_.emplace(operand_id.value(), tensor_desc).second) << "duplicate key";
      inputs_.push_back(operand_id.value());
    }
    for (const auto& result_id : ::llvm::enumerate(
             ::mlir::oneflow::user_op::ArgIds<mlir::OpTrait::AttrSizedResultSegments>(&op))) {
      user_op::NaiveTensorDesc tensor_desc{};
      auto result = op.getResult(result_id.index());
      if (auto rankedTensorType = result.getType().dyn_cast<mlir::RankedTensorType>()) {
        tensor_desc.set_shape(
            Shape{rankedTensorType.getShape().begin(), rankedTensorType.getShape().end()});
        tensor_desc.set_data_type(
            mlir::oneflow::support::GetDataTypeFromMLIRType(rankedTensorType.getElementType()));
        // TODO: set stride
        // TODO: set is_dynamic
      } else {
        LOG(FATAL) << "Unranked tensor type not supported";
      }
      CHECK(arg2tensor_desc_.emplace(result_id.value(), tensor_desc).second) << "duplicate key";
      outputs_.push_back(result_id.value());
    }
    auto dev_tag = mlir::OpTrait::IsOpConfCompatible<void>::getDeviceTag(&op);
    if (dev_tag == "cpu") {
      device_type_ = DeviceType::kCPU;
    } else if (dev_tag == "cuda") {
      device_type_ = DeviceType::kCUDA;
    } else {
      LOG(FATAL) << "Unsupported device tag: " << dev_tag.str();
    }
  }

  ~KernelLaunchOpKernelRegContext() = default;
  DeviceType device_type() const override { return device_type_; }
  const ParallelContext& parallel_ctx() const override {
    TODO() << "create parallel_ctx from op in mlir";
    ParallelContext* parallel_ctx = nullptr;
    return *parallel_ctx;
  }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) {
      LOG(FATAL) << "TensorDesc not found for arg_name: " << arg_name << " index: " << index;
    }
    return &(it->second);
  }
  const ArgVec& inputs() const override { return inputs_; }
  const ArgVec& outputs() const override { return outputs_; }

  const user_op::UserOpConfWrapper& user_op_conf() const override {
    TODO() << "get user op conf from op in mlir";
    OperatorConf user_op_conf;
    return user_op::UserOpConfWrapper(std::make_shared<OperatorConf>(user_op_conf));
  }

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4Name(attr_name);
  }

  ::mlir::Operation* GetOp() const { return op_; }

 private:
  ::mlir::func::FuncOp func_op_;
  ::mlir::ModuleOp owned_module_;
  ::mlir::Operation* op_;
  DeviceType device_type_ = DeviceType::kInvalidDevice;
  std::unordered_map<mlir::oneflow::user_op::ArgID, user_op::NaiveTensorDesc> arg2tensor_desc_{};
  ArgVec inputs_;
  ArgVec outputs_;
};

class KernelLaunchComputeContext final : public user_op::KernelComputeContext {
 public:
  explicit KernelLaunchComputeContext(std::shared_ptr<KernelLaunchOpKernelRegContext> reg,
                                      KernelComputeContext* comp)
      : reg_ctx_(std::move(reg)), comp_ctx_(comp) {}
  ~KernelLaunchComputeContext() = default;

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return reg_ctx_->TensorDesc4ArgNameAndIndex(arg_name, index);
  }

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    auto op = reg_ctx_->GetOp();
    LOG(ERROR) << "here";
    auto id = std::make_pair(arg_name, index);
    for (const auto& operand_id : ::llvm::enumerate(
             ::mlir::oneflow::user_op::ArgIds<mlir::OpTrait::AttrSizedOperandSegments>(op))) {
      if (operand_id.value() == id) {
        if (auto arg = op->getOperand(operand_id.index()).dyn_cast<mlir::BlockArgument>()) {
          return comp_ctx_->Tensor4ArgNameAndIndex("in", arg.getArgNumber());
        }
      }
    }
    for (const auto& result_id : ::llvm::enumerate(
             ::mlir::oneflow::user_op::ArgIds<mlir::OpTrait::AttrSizedResultSegments>(op))) {
      if (result_id.value() == id) {
        auto value = op->getResult(result_id.index());
        if (value.hasOneUse()) {
          mlir::Operation* first_user = value.use_begin()->getOwner();
          if (auto ret = llvm::dyn_cast_or_null<mlir::func::ReturnOp>(first_user)) {
            return comp_ctx_->Tensor4ArgNameAndIndex("out",
                                                     value.getUses().begin()->getOperandNumber());
          }
        }
      }
    }
    LOG(FATAL) << "Not supported";
  }

  ep::Stream* stream() override { return comp_ctx_->stream(); }

  DeviceType device_type() const override { return reg_ctx_->device_type(); }
  const ParallelContext& parallel_ctx() const override { return comp_ctx_->parallel_ctx(); }

  const ArgVec& inputs() const override { return reg_ctx_->inputs(); }
  const ArgVec& outputs() const override { return reg_ctx_->outputs(); }

  const user_op::UserOpConfWrapper& user_op_conf() const override {
    return reg_ctx_->user_op_conf();
  }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4Name(attr_name);
  }
  std::shared_ptr<KernelLaunchOpKernelRegContext> reg_ctx_;
  KernelComputeContext* comp_ctx_ = nullptr;
  std::unordered_map<mlir::oneflow::user_op::ArgID, user_op::Tensor*> tensor_desc_{};
};

class OKLRegContext final {
  static mlir::DialectRegistry GetRegistry() {
    mlir::DialectRegistry registry;
    registry.insert<mlir::LLVM::LLVMDialect, mlir::oneflow::OneFlowDialect, mlir::func::FuncDialect,
                    mlir::okl::OKLDialect>();
    mlir::registerLLVMDialectTranslation(registry);
    return registry;
  }

 public:
  explicit OKLRegContext(const char* mlir_asm) : mlir_ctx_(GetRegistry()) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_asm, &mlir_ctx_);
    if (!module) {
      LOG(ERROR) << "Fail to load mlir assembly";
      exit(1);
    }

    reg_ctx_ = std::make_shared<oneflow::KernelLaunchOpKernelRegContext>(*module);
  };
  ~OKLRegContext() = default;

  oneflow::KernelLaunchComputeContext* BuildRunContext(
      oneflow::user_op::KernelComputeContext* compute_ctx) {
    // this will be gc after run jit
    return new oneflow::KernelLaunchComputeContext(reg_ctx_, compute_ctx);
  }

  const oneflow::user_op::OpKernel* BuildKernel(const char* op_name) {
    auto* res = CHECK_JUST(oneflow::user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(
                               op_name, *reg_ctx_))
                    ->create_fn();
    return res;
  }

  void Launch(oneflow::KernelLaunchComputeContext* run_ctx, oneflow::user_op::OpKernel* kernel) {
    auto* okl_ctx = dynamic_cast<oneflow::user_op::KernelComputeContext*>(run_ctx);
    kernel->Compute(okl_ctx);
  }

 private:
  mlir::MLIRContext mlir_ctx_;
  std::shared_ptr<oneflow::KernelLaunchOpKernelRegContext> reg_ctx_;
};

void DoCompute(user_op::KernelComputeContext* ctx) {
  auto kernel_func_name = "okl_func";
  // generate the mlir ctx for lowering oneflow.kernel_launch to llvm dialect
  mlir::DialectRegistry registry;
  registry.insert<mlir::LLVM::LLVMDialect, mlir::func::FuncDialect, mlir::oneflow::OneFlowDialect,
                  mlir::okl::OKLDialect>();
  mlir::registerLLVMDialectTranslation(registry);
  mlir::MLIRContext mlir_ctx(registry);
  // fetch mlir_asm from ctx
  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(ctx->Attr<std::string>("mlir_assembly"), &mlir_ctx);
  if (!module) {
    LOG(ERROR) << "Failed to fetch mlir_asm in wrap_func.";
    exit(1);
  }
  // lower oneflow.kernel_launch to llvm dialect
  // if (failed(mlir::oneflow::LowerKernelLaunchModuleToLLVM(*module))) {
  //   LOG(ERROR) << "Failed to lower oneflow.kernel_launch to llvm.";
  //   exit(1);
  // }
  // create and run jit engine with llvm ir
  auto engine = std::make_shared<JIT_Engine>(*module);
  engine->Run(kernel_func_name, (void*)ctx);
}

template<typename T>
class KernelLaunchCpuKernel final : public user_op::OpKernel {
 public:
  KernelLaunchCpuKernel() = default;
  ~KernelLaunchCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override { DoCompute(ctx); }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_KERNEL_LAUNCH_CPU_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("kernel_launch")                                                         \
      .SetCreateFn<KernelLaunchCpuKernel<dtype>>()                                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                           \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_KERNEL_LAUNCH_CPU_KERNEL(float)
REGISTER_KERNEL_LAUNCH_CPU_KERNEL(double)
REGISTER_KERNEL_LAUNCH_CPU_KERNEL(int32_t)
REGISTER_KERNEL_LAUNCH_CPU_KERNEL(int64_t)
#undef REGISTER_KERNEL_LAUNCH_CPU_KERNEL

template<typename T>
class KernelLaunchGpuKernel final : public user_op::OpKernel {
 public:
  KernelLaunchGpuKernel() = default;
  ~KernelLaunchGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override { DoCompute(ctx); }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_KERNEL_LAUNCH_GPU_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("kernel_launch")                                                         \
      .SetCreateFn<KernelLaunchGpuKernel<dtype>>()                                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                          \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_KERNEL_LAUNCH_GPU_KERNEL(float)
REGISTER_KERNEL_LAUNCH_GPU_KERNEL(double)
REGISTER_KERNEL_LAUNCH_GPU_KERNEL(int32_t)
REGISTER_KERNEL_LAUNCH_GPU_KERNEL(int64_t)

#undef REGISTER_KERNEL_LAUNCH_GPU_KERNEL

}  // namespace

}  // namespace oneflow

extern "C" {
// llvm.call build_reg_ctx(gep_global_str: llvm_ptr<i8>) -> llvm_ptr<i8>
void* build_reg_ctx(void* gep_global_str) {
  return new oneflow::OKLRegContext((const char*)gep_global_str);
}

void destroy_reg_ctx(void* reg_ctx) { delete (oneflow::OKLRegContext*)reg_ctx; }

void* build_run_ctx(void* reg_ctx, void* compute_ctx) {
  return (void*)((oneflow::OKLRegContext*)reg_ctx)
      ->BuildRunContext((oneflow::user_op::KernelComputeContext*)compute_ctx);
}

void destroy_run_ctx(void* reg_ctx) { delete (oneflow::KernelLaunchComputeContext*)reg_ctx; }

void* build_kernel(void* reg_ctx, void* op_name) {
  return (void*)((oneflow::OKLRegContext*)reg_ctx)->BuildKernel((const char*)op_name);
}

void launch(void* reg_ctx, void* run_ctx, void* kernel) {
  ((oneflow::OKLRegContext*)reg_ctx)
      ->Launch((oneflow::KernelLaunchComputeContext*)run_ctx, (oneflow::user_op::OpKernel*)kernel);
}
}  // extern "C"
