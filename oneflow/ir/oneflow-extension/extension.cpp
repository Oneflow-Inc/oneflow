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
#include "mlir/Parser.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "OneFlow/OneFlowDialect.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/ir/include/OneFlow/Passes.h"

namespace oneflow {

namespace {

REGISTER_USER_OP("mlir_jit")
    .Attr<std::string>("mlir_assembly")
    .InputWithMinimum("in", 0)
    .OutputWithMinimum("out", 0)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      // TODO: infer shape by extracting Ops from mlir_assembly
      CHECK_EQ(ctx->inputs().size(), 1);
      CHECK_EQ(ctx->outputs().size(), 1);
      const Shape& in_shape = ctx->InputShape("in", 0);
      Shape* out_shape = ctx->OutputShape("out", 0);
      *out_shape = in_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

template<DeviceType device_type, typename T>
class MlirJitKernel final : public user_op::OpKernel {
 public:
  MlirJitKernel() = default;
  ~MlirJitKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(ERROR) << "MlirJitKernel::Compute";
    LOG(ERROR) << ctx->Attr<std::string>("mlir_assembly");
    // TODO: extrac a function
    mlir::DialectRegistry registry;
    registry.insert<mlir::oneflow::OneFlowDialect, mlir::StandardOpsDialect,
                    mlir::memref::MemRefDialect, mlir::tosa::TosaDialect,
                    mlir::linalg::LinalgDialect>();
    mlir::registerLLVMDialectTranslation(registry);
    mlir::MLIRContext mlir_ctx(registry);
    mlir::OwningModuleRef module =
        mlir::parseSourceString<mlir::ModuleOp>(ctx->Attr<std::string>("mlir_assembly"), &mlir_ctx);
    llvm::InitializeNativeTarget();
    CHECK(mlir::succeeded(mlir::oneflow::Lower(&mlir_ctx, *module)))
        << "fail to lower OneFlow to LLVM";
    module->dump();
    auto jit = mlir::ExecutionEngine::create(*module);
    CHECK(jit) << "failed to create JIT exe engine";
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

// TODO: figure out if device and dtype are necessary for this op?
#define REGISTER_MLIR_JIT_KERNEL(device, dtype)                                                 \
  REGISTER_USER_KERNEL("mlir_jit")                                                              \
      .SetCreateFn<MlirJitKernel<device, dtype>>()                                              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_MLIR_JIT_KERNEL(DeviceType::kCPU, float)
REGISTER_MLIR_JIT_KERNEL(DeviceType::kCPU, double)
REGISTER_MLIR_JIT_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_MLIR_JIT_KERNEL(DeviceType::kCPU, int64_t)

#undef REGISTER_MLIR_JIT_KERNEL

}  // namespace

}  // namespace oneflow
