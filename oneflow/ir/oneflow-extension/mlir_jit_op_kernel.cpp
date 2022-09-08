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
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "OneFlow/OneFlowDialect.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/ir/include/OneFlow/Passes.h"
#include "oneflow/ir/include/OneFlow/Extension.h"

namespace oneflow {

namespace {

using OpaqueMemRefDescriptor = std::shared_ptr<void>;

template<unsigned N, typename T>
OpaqueMemRefDescriptor CreateMemRefDescriptor(user_op::Tensor* tensor) {
  using MemRefType = StridedMemRefType<const T, N>;
  auto desc = new MemRefType();
  *desc = mlir::detail::makeStridedMemRefDescriptor<N>(
      tensor->dptr<T>(), tensor->dptr<T>(),
      {tensor->shape_view().ptr(), tensor->shape_view().ptr() + tensor->shape_view().NumAxes()},
      {tensor->shape_view().ptr(), tensor->shape_view().ptr() + tensor->shape_view().NumAxes()});
  auto deleter = [](void const* data) {
    auto p = static_cast<MemRefType const*>(data);
    delete p;
  };
  return OpaqueMemRefDescriptor(desc, deleter);
}

template<unsigned N, typename T>
OpaqueMemRefDescriptor CreateMutMemRefDescriptor(user_op::Tensor* tensor) {
  using MemRefType = StridedMemRefType<T, N>;
  auto desc = new MemRefType();
  *desc = mlir::detail::makeStridedMemRefDescriptor<N>(
      tensor->mut_dptr<T>(), tensor->mut_dptr<T>(),
      {tensor->shape_view().ptr(), tensor->shape_view().ptr() + tensor->shape_view().NumAxes()},
      {tensor->shape_view().ptr(), tensor->shape_view().ptr() + tensor->shape_view().NumAxes()});
  auto deleter = [](void const* data) {
    auto p = static_cast<MemRefType const*>(data);
    delete p;
  };
  return OpaqueMemRefDescriptor(desc, deleter);
}

#define MAKE_STRIDED_MEM_REF_SWITCH_ENTRY(func_name, N, T) func_name<N, T>
DEFINE_STATIC_SWITCH_FUNC(OpaqueMemRefDescriptor, CreateMemRefDescriptor,
                          MAKE_STRIDED_MEM_REF_SWITCH_ENTRY, MAKE_NDIM_CTRV_SEQ(DIM_SEQ),
                          MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
DEFINE_STATIC_SWITCH_FUNC(OpaqueMemRefDescriptor, CreateMutMemRefDescriptor,
                          MAKE_STRIDED_MEM_REF_SWITCH_ENTRY, MAKE_NDIM_CTRV_SEQ(DIM_SEQ),
                          MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
#undef MAKE_STRIDED_MEM_REF_SWITCH_ENTRY

std::string GetMLIRCInterface(const std::string& func_name) {
  return std::string("_mlir_ciface_") + func_name;
}

llvm::SmallVector<OpaqueMemRefDescriptor> GetMLIRCInterfaceArgs(
    user_op::KernelComputeContext* ctx) {
  llvm::SmallVector<OpaqueMemRefDescriptor> args{};
  for (auto& pair : ctx->inputs()) {
    auto tensor = ctx->Tensor4ArgNameAndIndex(pair.first, pair.second);
    auto ref = SwitchCreateMemRefDescriptor(
        SwitchCase(tensor->shape_view().NumAxes(), tensor->data_type()), tensor);
    args.push_back(ref);
  }
  for (auto& pair : ctx->outputs()) {
    auto tensor = ctx->Tensor4ArgNameAndIndex(pair.first, pair.second);
    auto ref = SwitchCreateMutMemRefDescriptor(
        SwitchCase(tensor->shape_view().NumAxes(), tensor->data_type()), tensor);
    args.push_back(ref);
  }
  return args;
}

void WithMlirContext(
    user_op::KernelComputeContext* ctx, const llvm::SmallVector<llvm::StringRef, 4>& ext_libs,
    const std::function<mlir::OwningOpRef<mlir::ModuleOp>(mlir::MLIRContext* mlir_ctx)>& parse,
    const std::function<void(mlir::MLIRContext* mlir_ctx, mlir::ModuleOp module)>& lower) {
  mlir::DialectRegistry registry;
  registry
      .insert<mlir::oneflow::OneFlowDialect, mlir::func::FuncDialect, mlir::memref::MemRefDialect,
              mlir::tosa::TosaDialect, mlir::linalg::LinalgDialect>();
  mlir::registerLLVMDialectTranslation(registry);
  mlir::MLIRContext mlir_ctx(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(&mlir_ctx);
  CHECK(!!module) << "fail to parse MLIR, op: " << ctx->op_name();
  if (ParseBooleanFromEnv("ONEFLOW_MLIR_STDOUT", false)) { module->print(llvm::outs()); }
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  lower(&mlir_ctx, *module);
  if (ParseBooleanFromEnv("ONEFLOW_MLIR_STDOUT", false)) { module->print(llvm::outs()); }
  if (ParseBooleanFromEnv("ONEFLOW_MLIR_DUMP_IR", false)) {
    std::string mlir;
    llvm::raw_string_ostream os_mlir(mlir);
    module->print(os_mlir);
    TeePersistentLogStream::Create(JoinPath("jit", ctx->op_name() + ".mlir"))->Write(mlir);
  }

  mlir::ExecutionEngineOptions jitOptions;
  jitOptions.transformer = {};
  jitOptions.jitCodeGenOptLevel = llvm::None;
  jitOptions.sharedLibPaths = ext_libs;

  auto jit_or_error = mlir::ExecutionEngine::create(*module, jitOptions);
  CHECK(!!jit_or_error) << "failed to create JIT exe engine, "
                        << llvm::toString(jit_or_error.takeError());
  auto jit = std::move(jit_or_error.get());
  llvm::SmallVector<OpaqueMemRefDescriptor> args /* args must outlive JIT invocation */ =
      GetMLIRCInterfaceArgs(ctx);
  llvm::SmallVector<void*> packed_args{};
  for (auto& arg /* arg must be a reference*/ : args) { packed_args.push_back(&arg); }
  auto error = jit->invokePacked(GetMLIRCInterface(ctx->op_name()), packed_args);
  CHECK(!error) << "fail to invoke jit engine, error: " << llvm::toString(std::move(error));
}

template<typename T>
class MlirJitCpuKernel final : public user_op::OpKernel {
 public:
  MlirJitCpuKernel() = default;
  ~MlirJitCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    llvm::SmallVector<llvm::StringRef, 4> ext_libs(
        {SharedLibPaths()->begin(), SharedLibPaths()->end()});
    WithMlirContext(
        ctx, ext_libs,
        [&ctx](mlir::MLIRContext* mlir_ctx) {
          return mlir::parseSourceString<mlir::ModuleOp>(ctx->Attr<std::string>("mlir_assembly"),
                                                         mlir_ctx);
        },
        [](mlir::MLIRContext* mlir_ctx, mlir::ModuleOp module) {
          CHECK(mlir::succeeded(mlir::oneflow::LowerModuleToLLVM(mlir_ctx, module)))
              << "fail to lower OneFlow to LLVM";
        });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLIR_JIT_CPU_KERNEL(dtype)                                                     \
  REGISTER_USER_KERNEL("mlir_jit")                                                              \
      .SetCreateFn<MlirJitCpuKernel<dtype>>()                                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                           \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_MLIR_JIT_CPU_KERNEL(float)
REGISTER_MLIR_JIT_CPU_KERNEL(double)
REGISTER_MLIR_JIT_CPU_KERNEL(int32_t)
REGISTER_MLIR_JIT_CPU_KERNEL(int64_t)

#undef REGISTER_MLIR_JIT_CPU_KERNEL

#ifdef WITH_MLIR_CUDA_CODEGEN

template<typename T>
class MlirJitGpuKernel final : public user_op::OpKernel {
 public:
  MlirJitGpuKernel() = default;
  ~MlirJitGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    llvm::SmallVector<llvm::StringRef, 4> ext_libs(
        {SharedLibPaths()->begin(), SharedLibPaths()->end()});
    WithMlirContext(
        ctx, ext_libs,
        [&ctx](mlir::MLIRContext* mlir_ctx) {
          return mlir::parseSourceString<mlir::ModuleOp>(ctx->Attr<std::string>("mlir_assembly"),
                                                         mlir_ctx);
        },
        [](mlir::MLIRContext* mlir_ctx, mlir::ModuleOp module) {
          CHECK(mlir::succeeded(mlir::oneflow::LowerModuleToCUDALLVM(mlir_ctx, module)))
              << "fail to lower OneFlow to CUDA LLVM";
        });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLIR_JIT_GPU_KERNEL(dtype)                                                     \
  REGISTER_USER_KERNEL("mlir_jit")                                                              \
      .SetCreateFn<MlirJitGpuKernel<dtype>>()                                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                          \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_MLIR_JIT_GPU_KERNEL(float)
REGISTER_MLIR_JIT_GPU_KERNEL(double)
REGISTER_MLIR_JIT_GPU_KERNEL(int32_t)
REGISTER_MLIR_JIT_GPU_KERNEL(int64_t)

#undef REGISTER_MLIR_JIT_GPU_KERNEL

#endif  // WITH_MLIR_CUDA_CODEGEN

}  // namespace

}  // namespace oneflow
