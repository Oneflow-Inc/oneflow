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
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/ir/include/OneFlow/Passes.h"
#include "oneflow/ir/include/OneFlow/Extension.h"
#include "oneflow/core/framework/op_generated.h"

extern "C" {

void launch_kernel(void* ctx, std::string name) {
  auto res = ::oneflow::user_op::UserOpRegistryMgr::Get().GetAllOpRegistryResults();
  auto it = res.find(name);
  if (it == res.end()) return;
  // TODO: LookUp
  // auto instance = oneflow::Singleton<oneflow::user_op::KernelLaunchRegistry>::Get();
  // auto name = instance->getName(index);
  // auto kernel = instance->LookUp(name)();
  auto converted_ctx = (oneflow::user_op::KernelComputeContext*)ctx;
  // kernel->Compute(converted_ctx);
}

}  // extern "C"

namespace oneflow {

Maybe<void> KernelLaunchOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
  ;
}

Maybe<void> KernelLaunchOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

Maybe<void> KernelLaunchOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

Maybe<void> KernelLaunchOp::InferDataType(user_op::InferContext* ctx) { return Maybe<void>::Ok(); }

namespace {

// this context should support querying information about the kernel from representation in MLIR
class KernelLaunchOpKernelRegContext final : public user_op::KernelRegContext {
  using ArgVec = std::vector<std::pair<std::string, int32_t>>;

 public:
  explicit KernelLaunchOpKernelRegContext();
  ~KernelLaunchOpKernelRegContext() = default;
  DeviceType device_type() const override { return device_type_; }
  const ParallelContext& parallel_ctx() const override { return *parallel_ctx_; }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return nullptr;
  }
  const ArgVec& inputs() const override { return {}; }
  const ArgVec& outputs() const override { return {}; }

  const user_op::UserOpConfWrapper& user_op_conf() const override { return user_op_conf_; }

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4Name(attr_name);
  }

 private:
  DeviceType device_type_;
  const ParallelContext* parallel_ctx_;
  const user_op::UserOpConfWrapper user_op_conf_;
};

template<typename T>
class KernelLaunchCpuKernel final : public user_op::OpKernel {
 public:
  KernelLaunchCpuKernel() = default;
  ~KernelLaunchCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    llvm::SmallVector<llvm::StringRef, 4> ext_libs(
        {SharedLibPaths()->begin(), SharedLibPaths()->end()});
    // TODO
    KernelLaunchOpKernelRegContext reg_ctx;
    const auto* res =
        CHECK_JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult("relu", reg_ctx));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  ::mlir::oneflow::KernelLaunchOp op_;
  ::mlir::ModuleOp owned_module_;
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
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // TODO
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
