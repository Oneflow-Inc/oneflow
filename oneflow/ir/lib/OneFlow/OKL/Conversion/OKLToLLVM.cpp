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
#include "OneFlow/OKL/OKLOps.h"
#include "OneFlow/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <string>

namespace mlir {

namespace okl {

template<typename Wrap, typename T>
ModuleOp GetModuleOpFromJobBodyOp(T op) {
  auto parent_func_op = op->template getParentOfType<Wrap>();
  if (!parent_func_op) { return nullptr; }
  return parent_func_op->template getParentOfType<ModuleOp>();
}

LLVM::GlobalOp DeclareOrGetGlobalString(::mlir::PatternRewriter& rewriter, ModuleOp* module,
                                        StringRef key, StringRef val, bool increase = false) {
  auto create = [&]() -> LLVM::GlobalOp {
    OpBuilder::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module->getBody());
    auto type = LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), val.size());
    return rewriter.create<LLVM::GlobalOp>(rewriter.getUnknownLoc(), type, /*isConstant=*/true,
                                           LLVM::Linkage::Internal, key,
                                           rewriter.getStringAttr(val),
                                           /*alignment=*/0);
  };
  LLVM::GlobalOp global;
  if ((global = module->lookupSymbol<LLVM::GlobalOp>(key))) {
    if (increase) {
      int idx = 1;
      do key = StringRef(key.str() + std::to_string(idx++));
      while ((global = module->lookupSymbol<LLVM::GlobalOp>(key)));
      return create();
    }
  } else {
    return create();
  }
  return global;
}

LLVM::LLVMPointerType GetPtrType(::mlir::PatternRewriter& rewriter) {
  return LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8));
}

LLVM::LLVMFuncOp DeclareBuildRunContext(::mlir::PatternRewriter& rewriter, ModuleOp* module) {
  auto func_name = "build_run_ctx";
  LLVM::LLVMFuncOp func;
  if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(func_name))) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module->getBody());
    auto void_type = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto func_type =
        LLVM::LLVMFunctionType::get(void_type, {GetPtrType(rewriter), GetPtrType(rewriter)}, false);
    func = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name, func_type,
                                             LLVM::Linkage::External);
    func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
  }
  return func;
}

LLVM::LLVMFuncOp DeclareBuildRegContext(::mlir::PatternRewriter& rewriter, ModuleOp* module) {
  auto func_name = "build_reg_ctx";
  LLVM::LLVMFuncOp func;
  if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(func_name))) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module->getBody());
    auto void_type = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto func_type =
        LLVM::LLVMFunctionType::get(void_type, {GetPtrType(rewriter), GetPtrType(rewriter)}, false);
    func = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name, func_type,
                                             LLVM::Linkage::External);
    func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
  }
  return func;
}

LLVM::LLVMFuncOp DeclareBuildKernel(::mlir::PatternRewriter& rewriter, ModuleOp* module) {
  auto func_name = "build_kernel";
  LLVM::LLVMFuncOp func;
  if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(func_name))) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module->getBody());
    auto void_type = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto func_type =
        LLVM::LLVMFunctionType::get(void_type, {GetPtrType(rewriter), GetPtrType(rewriter)}, false);
    func = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name, func_type,
                                             LLVM::Linkage::External);
    func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
  }
  return func;
}

LLVM::LLVMFuncOp DeclareBuildLaunch(::mlir::PatternRewriter& rewriter, ModuleOp* module) {
  auto func_name = "build_launch";
  LLVM::LLVMFuncOp func;
  if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(func_name))) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module->getBody());
    auto void_type = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto func_type =
        LLVM::LLVMFunctionType::get(void_type, {GetPtrType(rewriter), GetPtrType(rewriter)}, false);
    func = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name, func_type,
                                             LLVM::Linkage::External);
    func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
  }
  return func;
}

// create okl.run_ctx(*reg_ctx, *compute_ctx)
struct RunContextOpLowering final : public OpConversionPattern<RunContextOp> {
 public:
  using OpConversionPattern<RunContextOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(RunContextOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op->getLoc();
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto compute_ctx = func.getArgument(0);
    auto module = GetModuleOpFromJobBodyOp<LLVM::LLVMFuncOp>(op);

    auto reg_ctx = op->getOpOperand(0).get();
    auto build_run_ctx = DeclareBuildRunContext(rewriter, &module);
    auto res = rewriter.create<func::CallOp>(loc, build_run_ctx, ValueRange{compute_ctx, reg_ctx})
                   ->getResults();
    rewriter.replaceOp(op, ValueRange{res});
    return success();
  }
};

// create okl.reg_ctx(StringAttr: assembly)
struct RegContextOpLowering final : public OpConversionPattern<RegContextOp> {
 public:
  using OpConversionPattern<RegContextOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(RegContextOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op->getLoc();
    auto mlir_asm = op.mlir_assembly();

    op->getParentOfType<LLVM::LLVMFuncOp>();
    auto module = GetModuleOpFromJobBodyOp<LLVM::LLVMFuncOp>(op);
    auto global_str = DeclareOrGetGlobalString(rewriter, &module, "mlir_asm", mlir_asm, true);

    Value addr = rewriter.create<LLVM::AddressOfOp>(loc, global_str);
    Value cst0 =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getIndexAttr(0));
    auto gep =
        rewriter
            .create<LLVM::GEPOp>(loc, GetPtrType(rewriter), addr, ArrayRef<Value>({cst0, cst0}))
            ->getResult(0);
    auto build_reg_ctx = DeclareBuildRegContext(rewriter, &module);
    auto res = rewriter.create<func::CallOp>(loc, build_reg_ctx, ValueRange{gep})->getResults();
    rewriter.replaceOp(op, ValueRange{res});
    return success();
  }
};

// create okl.kernel(StringAttr: op_type_name, *reg_ctx)
struct KernelOpLowering final : public OpConversionPattern<KernelOp> {
 public:
  using OpConversionPattern<KernelOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(KernelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op->getLoc();
    auto module = GetModuleOpFromJobBodyOp<LLVM::LLVMFuncOp>(op);

    auto build_launch = DeclareBuildLaunch(rewriter, &module);
    auto op_type_name = op.op_type_name();

    auto global_str = DeclareOrGetGlobalString(rewriter, &module, op_type_name, op_type_name);
    Value addr = rewriter.create<LLVM::AddressOfOp>(loc, global_str);
    Value cst0 =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getIndexAttr(0));
    auto gep =
        rewriter
            .create<LLVM::GEPOp>(loc, GetPtrType(rewriter), addr, ArrayRef<Value>({cst0, cst0}))
            ->getResult(0);

    auto reg_ctx = op.reg_ctx();
    auto res =
        rewriter.create<func::CallOp>(loc, build_launch, ValueRange{gep, reg_ctx})->getResults();
    rewriter.replaceOp(op, ValueRange{res});
    return success();
  }
};

// create okl.launch(*run_ctx, *kernel)
struct LaunchOpLowering final : public OpConversionPattern<LaunchOp> {
 public:
  using OpConversionPattern<LaunchOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(LaunchOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op->getLoc();
    auto module = GetModuleOpFromJobBodyOp<LLVM::LLVMFuncOp>(op);

    auto build_launch = DeclareBuildLaunch(rewriter, &module);
    auto run_ctx = op.run_ctx();
    auto kernel = op.kernel();
    auto res =
        rewriter.create<func::CallOp>(loc, build_launch, ValueRange{run_ctx, kernel})->getResults();
    rewriter.replaceOp(op, ValueRange{res});
    return success();
  }
};

namespace {
struct OKLToLLVMPass : public LowerOneFlowToTosaPassBase<OKLToLLVMPass> {
  void runOnOperation() override;
};
}  // namespace

std::unique_ptr<Pass> createOKLToLLVMPass() { return std::make_unique<OKLToLLVMPass>(); }

void OKLToLLVMPass::runOnOperation() {
  MLIRContext* context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<okl::OKLDialect>();

  TypeConverter typeConverter;
  RewritePatternSet patterns(context);

  patterns.add<KernelOpLowering, LaunchOpLowering, RegContextOpLowering, RunContextOpLowering>(
      typeConverter, context);
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
    getOperation()->emitError("Failed to lower OKL to LLVM");
  }
}

}  // namespace okl

}  // namespace mlir
