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
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "OneFlow/OneFlowDialect.h"
#include "oneflow/core/common/switch_func.h"
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
      CHECK_EQ(ctx->inputs().size(), 2);
      CHECK_EQ(ctx->outputs().size(), 1);
      const Shape& in_shape = ctx->InputShape("in", 0);
      Shape* out_shape = ctx->OutputShape("out", 0);
      *out_shape = in_shape;
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 1);
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

using OpaqueMemRefDescriptor = std::shared_ptr<void>;

template<unsigned N, typename T>
OpaqueMemRefDescriptor CreateMemRefDescriptor(user_op::Tensor* tensor) {
  using MemRefType = StridedMemRefType<const T, N>;
  auto desc = new MemRefType();
  *desc = mlir::detail::makeStridedMemRefDescriptor<N>(
      tensor->dptr<T>(), tensor->dptr<T>(),
      {tensor->shape().ptr(), tensor->shape().ptr() + tensor->shape().NumAxes()},
      {tensor->shape().ptr(), tensor->shape().ptr() + tensor->shape().NumAxes()});
  auto deleter = [](void const* data) {
    auto p = static_cast<MemRefType const*>(data);
    delete p;
  };

  OpaqueMemRefDescriptor ret(desc, deleter);
  return ret;
}

template<unsigned N, typename T>
OpaqueMemRefDescriptor CreateMutMemRefDescriptor(user_op::Tensor* tensor) {
  using MemRefType = StridedMemRefType<T, N>;
  auto desc = new MemRefType();
  *desc = mlir::detail::makeStridedMemRefDescriptor<N>(
      tensor->mut_dptr<T>(), tensor->mut_dptr<T>(),
      {tensor->shape().ptr(), tensor->shape().ptr() + tensor->shape().NumAxes()},
      {tensor->shape().ptr(), tensor->shape().ptr() + tensor->shape().NumAxes()});
  auto deleter = [](void const* data) {
    auto p = static_cast<MemRefType const*>(data);
    delete p;
  };

  OpaqueMemRefDescriptor ret(desc, deleter);
  return ret;
}

#define MAKE_STRIDED_MEM_REF_SWITCH_ENTRY(func_name, N, T) func_name<N, T>
DEFINE_STATIC_SWITCH_FUNC(OpaqueMemRefDescriptor, CreateMemRefDescriptor,
                          MAKE_STRIDED_MEM_REF_SWITCH_ENTRY, MAKE_NDIM_CTRV_SEQ(DIM_SEQ),
                          MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
DEFINE_STATIC_SWITCH_FUNC(OpaqueMemRefDescriptor, CreateMutMemRefDescriptor,
                          MAKE_STRIDED_MEM_REF_SWITCH_ENTRY, MAKE_NDIM_CTRV_SEQ(DIM_SEQ),
                          MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
#undef MAKE_STRIDED_MEM_REF_SWITCH_ENTRY

template<DeviceType device_type, typename T>
class MlirJitKernel final : public user_op::OpKernel {
 public:
  MlirJitKernel() = default;
  ~MlirJitKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(ERROR) << "MlirJitKernel::Compute";
    LOG(ERROR) << ctx->Attr<std::string>("mlir_assembly");
    // TODO: extract a function
    mlir::DialectRegistry registry;
    registry.insert<mlir::oneflow::OneFlowDialect, mlir::StandardOpsDialect,
                    mlir::memref::MemRefDialect, mlir::tosa::TosaDialect,
                    mlir::linalg::LinalgDialect>();
    mlir::registerLLVMDialectTranslation(registry);
    mlir::MLIRContext mlir_ctx(registry);
    mlir::OwningModuleRef module =
        mlir::parseSourceString<mlir::ModuleOp>(ctx->Attr<std::string>("mlir_assembly"), &mlir_ctx);
    CHECK(!!module);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    CHECK(mlir::succeeded(mlir::oneflow::Lower(&mlir_ctx, *module)))
        << "fail to lower OneFlow to LLVM";
    if (std::getenv("ONEFLOW_MLIR_STDOUT") != nullptr) { module->print(llvm::outs()); }
    auto jit_or_error = mlir::ExecutionEngine::create(*module);
    CHECK(!!jit_or_error) << "failed to create JIT exe engine, "
                          << llvm::toString(jit_or_error.takeError());
    user_op::Tensor* in_0 = ctx->Tensor4ArgNameAndIndex("in", 0);
    // TODO: extract a function
    auto ref_in_0 =
        SwitchCreateMemRefDescriptor(SwitchCase(in_0->shape().NumAxes(), in_0->data_type()), in_0);
    user_op::Tensor* in_1 = ctx->Tensor4ArgNameAndIndex("in", 1);
    auto ref_in_1 =
        SwitchCreateMemRefDescriptor(SwitchCase(in_1->shape().NumAxes(), in_1->data_type()), in_1);
    user_op::Tensor* out_0 = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto ref_out_0 = SwitchCreateMutMemRefDescriptor(
        SwitchCase(out_0->shape().NumAxes(), out_0->data_type()), out_0);
    auto jit = std::move(jit_or_error.get());
    auto error = jit->invoke(ctx->op_name(), ref_in_0, ref_in_1, ref_out_0);
    CHECK(!error) << "fail to invoke jit engine, error: " << llvm::toString(std::move(error));
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
