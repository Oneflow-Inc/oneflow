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

#include "OneFlow/OKL/Conversion/Conversion.h"
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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/framework/op_generated.h"
#include "OneFlow/OKL/Kernel/JITOpInfer.h"
#include "OneFlow/OKL/Kernel/JITEngine.h"
#include "OneFlow/OKL/Kernel/LauncherState.h"
#include "OneFlow/OKL/Kernel/TmpBufferManager.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

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
  return ir::jit::SetTensorDataType(ctx);
}

namespace {

using namespace oneflow::okl;

template<typename T>
class KernelLaunchKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  KernelLaunchKernel() = default;
  ~KernelLaunchKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    // use ctx to create module, reg_ctx and fn;
    std::shared_ptr<user_op::OpKernelState> res(new LauncherState(ctx));
    return res;
  }

  bool IsCudaGraphSupported(user_op::KernelInitContext* ctx,
                            user_op::OpKernelState* state) const override {
    return dynamic_cast<LauncherState*>(state)->IsCudaGraphSupported(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* okl_state = dynamic_cast<LauncherState*>(state);
    okl_state->DoCompute(ctx);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_KERNEL_LAUNCH_CPU_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("kernel_launch")                                                         \
      .SetCreateFn<KernelLaunchKernel<dtype>>()                                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                           \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))        \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                       \
        return oneflow::okl::TmpBufferManager::InferTmpSize(ctx);                               \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_KERNEL_LAUNCH_CPU_KERNEL(float)
REGISTER_KERNEL_LAUNCH_CPU_KERNEL(double)
REGISTER_KERNEL_LAUNCH_CPU_KERNEL(int32_t)
REGISTER_KERNEL_LAUNCH_CPU_KERNEL(int64_t)
#undef REGISTER_KERNEL_LAUNCH_CPU_KERNEL

#define REGISTER_KERNEL_LAUNCH_GPU_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("kernel_launch")                                                         \
      .SetCreateFn<KernelLaunchKernel<dtype>>()                                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                          \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))        \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                       \
        return oneflow::okl::TmpBufferManager::InferTmpSize(ctx);                               \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_KERNEL_LAUNCH_GPU_KERNEL(float)
REGISTER_KERNEL_LAUNCH_GPU_KERNEL(double)
REGISTER_KERNEL_LAUNCH_GPU_KERNEL(int8_t)
REGISTER_KERNEL_LAUNCH_GPU_KERNEL(int32_t)
REGISTER_KERNEL_LAUNCH_GPU_KERNEL(int64_t)

#if CUDA_VERSION >= 11000
REGISTER_KERNEL_LAUNCH_GPU_KERNEL(half)
REGISTER_KERNEL_LAUNCH_GPU_KERNEL(nv_bfloat16)
#endif

#undef REGISTER_KERNEL_LAUNCH_GPU_KERNEL

}  // namespace

}  // namespace oneflow
