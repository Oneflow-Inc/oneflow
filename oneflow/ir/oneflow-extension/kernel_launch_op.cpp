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
#include <sys/types.h>
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/UserOpReflection.h"
#include "OneFlow/Passes.h"
#include "OneFlow/Extension.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/kernel/blob_tensor_view.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/JITOpInfer.h"

namespace oneflow {
using TypeKernelLaunchArgs =
    std::tuple<oneflow::user_op::KernelComputeContext*, const oneflow::user_op::OpKernel*>;
}

extern "C" {

void kernel_launch(void* ctx, void* kernel_opaque) {
  auto kernel = (typename std::tuple_element_t<1, oneflow::TypeKernelLaunchArgs>)kernel_opaque;
  kernel->Compute((typename std::tuple_element_t<0, oneflow::TypeKernelLaunchArgs>)ctx);
}

}  // extern "C"

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
    TODO() << "get user op conf rom op in mlir";
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
  explicit KernelLaunchComputeContext(std::unique_ptr<KernelLaunchOpKernelRegContext> reg,
                                      KernelComputeContext* comp)
      : reg_ctx_(std::move(reg)), comp_ctx_(comp) {}
  ~KernelLaunchComputeContext() = default;

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return reg_ctx_->TensorDesc4ArgNameAndIndex(arg_name, index);
  }

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    auto op = reg_ctx_->GetOp();
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
  std::unique_ptr<KernelLaunchOpKernelRegContext> reg_ctx_;
  KernelComputeContext* comp_ctx_ = nullptr;
  std::unordered_map<mlir::oneflow::user_op::ArgID, user_op::Tensor*> tensor_desc_{};
};

namespace {

template<typename ArgsT, class... Args>
void runJIT(mlir::ModuleOp module_op, llvm::StringRef name, Args... args) {
  using Tuple = std::tuple<Args...>;
  static_assert(std::is_same<ArgsT, Tuple>::value, "args of jit function don't match");
  llvm::SmallVector<llvm::StringRef, 4> ext_libs(
      {SharedLibPaths()->begin(), SharedLibPaths()->end()});
  mlir::ExecutionEngineOptions jitOptions;
  jitOptions.transformer = {};
  jitOptions.jitCodeGenOptLevel = llvm::None;
  jitOptions.sharedLibPaths = ext_libs;

  auto jit_or_error = mlir::ExecutionEngine::create(module_op, jitOptions);
  CHECK(!!jit_or_error) << "failed to create JIT exe engine, "
                        << llvm::toString(jit_or_error.takeError());
  auto jit = std::move(jit_or_error.get());
  // ctx->op_name(), &kl_comp_ctx, kernel
  auto error = jit->invoke(name, args...);
  CHECK(!error) << "fail to invoke jit engine, error: " << llvm::toString(std::move(error));
}

void KernelLaunchCompute(user_op::KernelComputeContext* ctx,
                         const oneflow::user_op::OpKernel* kernel) {
  mlir::DialectRegistry registry;
  registry
      .insert<mlir::oneflow::OneFlowDialect, mlir::func::FuncDialect, mlir::memref::MemRefDialect,
              mlir::tosa::TosaDialect, mlir::linalg::LinalgDialect, mlir::LLVM::LLVMDialect>();
  mlir::registerLLVMDialectTranslation(registry);
  mlir::MLIRContext mlir_ctx(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      mlir::parseSourceString<mlir::ModuleOp>(ctx->Attr<std::string>("mlir_assembly"), &mlir_ctx);

  auto reg_ctx = std::make_unique<KernelLaunchOpKernelRegContext>(module_op.get());
  if (kernel == nullptr) {
    const user_op::OpKernelRegistryResult* res =
        CHECK_JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(
            reg_ctx->GetOp()->getName().stripDialect().str(), *reg_ctx));
    kernel = res->create_fn();
  }
  KernelLaunchComputeContext kl_comp_ctx(std::move(reg_ctx), ctx);

  if (failed(mlir::oneflow::LowerKernelLaunchModuleToLLVM(*module_op))) {
    LOG(ERROR) << "Fail lowering kernel launch Module to llvm ir";
    exit(1);
  }
  runJIT<TypeKernelLaunchArgs>(module_op.get(), ctx->op_name(),
                               dynamic_cast<user_op::KernelComputeContext*>(&kl_comp_ctx), kernel);
}
}  // namespace

template<typename T>
class KernelLaunchCpuKernel final : public user_op::OpKernel {
 public:
  KernelLaunchCpuKernel() = default;
  ~KernelLaunchCpuKernel() = default;

 private:
  const oneflow::user_op::OpKernel* kernel;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    KernelLaunchCompute(ctx, kernel);
  }
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
  const oneflow::user_op::OpKernel* kernel;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    KernelLaunchCompute(ctx, kernel);
  }
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
