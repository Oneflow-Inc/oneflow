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
#include "OneFlow/OKL/OKLTypes.h"
#include "OneFlow/OKL/passes.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <string>
#include <glog/logging.h>

namespace mlir {
namespace okl {

template<typename Wrap, typename T>
ModuleOp GetModuleOpFromJobBodyOp(T op) {
  auto parent_func_op = op->template getParentOfType<Wrap>();
  if (!parent_func_op) { return nullptr; }
  return parent_func_op->template getParentOfType<ModuleOp>();
}

// use this func to union the ptr type in this conversion phase.
LLVM::LLVMPointerType GetPtrType(::mlir::PatternRewriter& rewriter) {
  return LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8));
}

template<typename T>
std::string FetchName() {
  LOG(ERROR) << "Faile to get type name";
  exit(1);
}

template<>
std::string FetchName<FetchKernelOp>() {
  return "fetch_kernel";
}

template<>
std::string FetchName<FetchRegContextOp>() {
  return "fetch_reg_ctx";
}

template<>
std::string FetchName<FetchRunContextOp>() {
  return "fetch_run_ctx";
}

template<typename T>
LLVM::LLVMFuncOp DeclareFetchPtr(::mlir::PatternRewriter& rewriter, ModuleOp* module) {
  LLVM::LLVMFuncOp func;
  auto func_name = FetchName<T>();
  if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(func_name))) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module->getBody());

    auto func_type = LLVM::LLVMFunctionType::get(
        {GetPtrType(rewriter)}, {GetPtrType(rewriter), rewriter.getI64Type()}, false);
    func = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name, func_type,
                                             LLVM::Linkage::External);
    func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
  }
  return func;
}

// lower !okl.launch to !llvm.call
struct LaunchOpLowering final : public OpConversionPattern<LaunchOp> {
  // raw: create okl.launch(*run_ctx, *kernel) -> llvm_ptr<i8>
  // dst: llvm.call launch(run_ctx: llvm_ptr<i8>, kernel: llvm_ptr<i8>)
  static LLVM::LLVMFuncOp DeclareBuildLaunch(::mlir::PatternRewriter& rewriter, ModuleOp* module) {
    auto func_name = "launch";
    LLVM::LLVMFuncOp func;
    if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(func_name))) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module->getBody());

      auto void_type = LLVM::LLVMVoidType::get(rewriter.getContext());
      auto func_type = LLVM::LLVMFunctionType::get(
          void_type, {GetPtrType(rewriter), GetPtrType(rewriter)}, false);
      func = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name, func_type,
                                               LLVM::Linkage::External);
      func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
    }
    return func;
  }

 public:
  static BlockAndValueMapping mapping;
  using OpConversionPattern<LaunchOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(LaunchOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto module = GetModuleOpFromJobBodyOp<func::FuncOp>(op);
    if (!module) {
      op->emitError("Failed to lowering llvm call because of op is not in a module");
      exit(1);
    };

    auto build_launch = DeclareBuildLaunch(rewriter, &module);
    auto run_ctx = mapping.lookup(op.run_ctx());
    auto kernel = mapping.lookup(op.kernel());

    rewriter.create<LLVM::CallOp>(op->getLoc(), build_launch, ValueRange{run_ctx, kernel});
    rewriter.eraseOp(op);
    return success();
  }
};
BlockAndValueMapping LaunchOpLowering::mapping;

// lower !okl.fetch_from_{T} to !llvm.call
template<typename T>
struct FetchOpLowering final : public OpConversionPattern<T> {
  using OpConversionPattern<T>::OpConversionPattern;
  using OpAdaptor = typename T::Adaptor;
  LogicalResult matchAndRewrite(T op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const override {
    auto module = GetModuleOpFromJobBodyOp<func::FuncOp>(op);
    if (!module) {
      op->emitError("Failed to lowering llvm call because of op is not in a module");
      exit(1);
    };

    auto fetch_ctx = DeclareFetchPtr<T>(rewriter, &module);
    auto launcher_ctx = op->template getParentOfType<func::FuncOp>().getBody().getArgument(0);
    auto index = rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI64Type(),
                                                   rewriter.getIndexAttr(op.index()));
    auto new_op =
        rewriter.create<LLVM::CallOp>(op->getLoc(), fetch_ctx, ValueRange{launcher_ctx, index});
    rewriter.replaceOp(op, new_op.getResults());
    LaunchOpLowering::mapping.map(op->getResult(0), new_op.getResult(0));
    return success();
  }
};


// change func.func(!okl.launcher_ctx) -> func.func(!llvm.ptr<i8>) { unrealized_conversion_cast(): !llvm.ptr<i8> -> !okl.launcher_ctx }
struct RewriteFunctionArgsPattern final : public mlir::OpRewritePattern<func::FuncOp> {
  static LogicalResult ConvertLauncherToLLVMPtr(func::FuncOp op, mlir::PatternRewriter& rewriter) {
    auto func_type = rewriter.getFunctionType({GetPtrType(rewriter)}, {});
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getSymName(), func_type);
    func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
    BlockAndValueMapping bvm;
    op.getRegion().cloneInto(&func.getRegion(), bvm);
    auto& block = func.getBody().getBlocks().front();
    auto launcher_ctx = block.getArgument(0);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    auto cast_op = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), launcher_ctx.getType(),
                                                               launcher_ctx);
    launcher_ctx.setType(GetPtrType(rewriter));
    launcher_ctx.replaceAllUsesExcept(cast_op->getResult(0), {cast_op});
    rewriter.setInsertionPointToEnd(&block);
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(&block.back(), ValueRange());
    rewriter.eraseOp(op);
    return success();
  }

 public:
  explicit RewriteFunctionArgsPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    if (op.getNumArguments() == 1
        && op.getArgumentTypes().begin()->isa<okl::LauncherContextType>()) {
      return ConvertLauncherToLLVMPtr(op, rewriter);
    }
    return success();
  }
};

namespace {
struct LowerLauncherToLLVMPtrPass
    : public LowerLauncherToLLVMPtrPassBase<LowerLauncherToLLVMPtrPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<okl::OKLDialect>();
  }
};

struct LowerOKLToLLVMCallPass : public LowerOKLToLLVMCallPassBase<LowerOKLToLLVMCallPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<okl::OKLDialect>();
  }
};
}  // namespace

std::unique_ptr<Pass> createLowerOKLToLLVMCallPass() {
  return std::make_unique<LowerOKLToLLVMCallPass>();
}
std::unique_ptr<Pass> createLowerLauncherToLLVMPtrPass() {
  return std::make_unique<LowerLauncherToLLVMPtrPass>();
}

void LowerLauncherToLLVMPtrPass::runOnOperation() {
  MLIRContext* context = &getContext();
  RewritePatternSet patterns(context);

  patterns.add<RewriteFunctionArgsPattern>(context);

  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void LowerOKLToLLVMCallPass::runOnOperation() {
  MLIRContext* context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<okl::OKLDialect>();

  auto llvm_ptr_type = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  TypeConverter typeConverter;
  typeConverter.addConversion([&](mlir::okl::LauncherContextType type) { return llvm_ptr_type; });
  typeConverter.addConversion([&](mlir::okl::RegContextType type) { return llvm_ptr_type; });
  typeConverter.addConversion([&](mlir::okl::RunContextType type) { return llvm_ptr_type; });
  typeConverter.addConversion([&](mlir::okl::KernelType type) { return llvm_ptr_type; });

  RewritePatternSet patterns(context);

  patterns.add<FetchOpLowering<FetchRegContextOp>, FetchOpLowering<FetchRunContextOp>,
               FetchOpLowering<FetchKernelOp>, LaunchOpLowering>(typeConverter, context);

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
    getOperation()->emitError("Failed to lower OKL to LLVM Call");
  }
}

}  // namespace okl
}  // namespace mlir
