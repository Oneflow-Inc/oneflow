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
#include "OneFlow/OKL/passes.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <string>

namespace mlir {

namespace okl {

// template<typename Wrap, typename T>
// ModuleOp GetModuleOpFromJobBodyOp(T op) {
//   auto parent_func_op = op->template getParentOfType<Wrap>();
//   if (!parent_func_op) { return nullptr; }
//   return parent_func_op->template getParentOfType<ModuleOp>();
// }

// // use this func to union the ptr type in this conversion phase.
// LLVM::LLVMPointerType GetPtrType(::mlir::PatternRewriter& rewriter) {
//   return LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8));
// }

// // get gep op as a ptr value from global field.
// LLVM::GEPOp GetGepOpFromGlobal(::mlir::PatternRewriter& rewriter, ModuleOp* module,
//                                LLVM::GlobalOp* global) {
//   auto loc = rewriter.getUnknownLoc();
//   Value addr = rewriter.create<LLVM::AddressOfOp>(loc, *global);
//   Value cst0 =
//       rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getIndexAttr(0));
//   return rewriter.create<LLVM::GEPOp>(loc, GetPtrType(rewriter), addr,
//                                       ArrayRef<Value>({cst0, cst0}));
// }

// // declare key:string = vale in llvm ir global scope.
// LLVM::GlobalOp DeclareOrGetGlobalString(::mlir::PatternRewriter& rewriter, ModuleOp* module,
//                                         StringRef key, StringRef val, bool increase = false) {
//   auto create = [&]() -> LLVM::GlobalOp {
//     OpBuilder::InsertionGuard insertGuard(rewriter);
//     rewriter.setInsertionPointToStart(module->getBody());
//     auto type = LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), val.size());
//     return rewriter.create<LLVM::GlobalOp>(rewriter.getUnknownLoc(), type, /*isConstant=*/true,
//                                            LLVM::Linkage::Internal, key,
//                                            rewriter.getStringAttr(val),
//                                            /*alignment=*/0);
//   };

//   LLVM::GlobalOp global;
//   if ((global = module->lookupSymbol<LLVM::GlobalOp>(key))) {
//     if (increase) {
//       int idx = 1;
//       do key = StringRef(key.str() + std::to_string(idx++));
//       while ((global = module->lookupSymbol<LLVM::GlobalOp>(key)));
//       return create();
//     }
//     return global;
//   }
//   return create();
// }

// // create okl.reg_ctx(StringAttr: assembly)
// struct RegContextOpLowering final : public OpConversionPattern<RegContextOp> {
//   // raw: create okl.reg_ctx(StringAttr: assembly) -> llvm_ptr<i8>
//   // dst: llvm.call build_reg_ctx(gep_global_str: llvm_ptr<i8>) -> llvm_ptr<i8>
//   static StringRef DeclareBuildRegContext(::mlir::PatternRewriter& rewriter, ModuleOp* module) {
//     auto func_name = "build_reg_ctx";
//     LLVM::LLVMFuncOp func;
//     if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(func_name))) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(module->getBody());

//       auto func_type =
//           LLVM::LLVMFunctionType::get(GetPtrType(rewriter), {GetPtrType(rewriter)}, false);
//       func = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name, func_type,
//                                                LLVM::Linkage::External);
//       func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
//     }
//     return func_name;
//   }

//  public:
//   using OpConversionPattern<RegContextOp>::OpConversionPattern;
//   LogicalResult matchAndRewrite(RegContextOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter& rewriter) const override {
//     auto mlir_asm = op.mlir_assembly();

//     op->getParentOfType<func::FuncOp>();
//     auto module = GetModuleOpFromJobBodyOp<func::FuncOp>(op);

//     auto global_str = DeclareOrGetGlobalString(rewriter, &module, "mlir_asm", mlir_asm, true);
//     auto gep = GetGepOpFromGlobal(rewriter, &module, &global_str);

//     auto build_reg_ctx = DeclareBuildRegContext(rewriter, &module);

//     rewriter
//         .replaceOpWithNewOp<LLVM::CallOp>(op, op->getResultTypes(), build_reg_ctx, ValueRange{gep})
//         ->getResult(0);
//     return success();
//   }
// };

// // create okl.run_ctx(*reg_ctx, *compute_ctx)
// struct RunContextOpLowering final : public OpConversionPattern<RunContextOp> {
//   // raw: create okl.run_ctx(*reg_ctx, *compute_ctx) -> llvm_ptr<i8>
//   // dst: llvm.call build_run_ctx(reg_ctx: llvm_ptr<i8>, compute_ctx: llvm_ptr<i8>) -> llvm_ptr<i8>
//   static LLVM::LLVMFuncOp DeclareBuildRunContext(::mlir::PatternRewriter& rewriter,
//                                                  ModuleOp* module) {
//     auto func_name = "build_run_ctx";
//     LLVM::LLVMFuncOp func;
//     if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(func_name))) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(module->getBody());

//       auto func_type = LLVM::LLVMFunctionType::get(
//           GetPtrType(rewriter), {GetPtrType(rewriter), GetPtrType(rewriter)}, false);
//       func = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name, func_type,
//                                                LLVM::Linkage::External);
//       func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
//     }
//     return func;
//   }

//  public:
//   using OpConversionPattern<RunContextOp>::OpConversionPattern;
//   LogicalResult matchAndRewrite(RunContextOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter& rewriter) const override {
//     auto func = op->getParentOfType<func::FuncOp>();

//     auto module = GetModuleOpFromJobBodyOp<func::FuncOp>(op);
//     auto reg_ctx = op.reg_ctx();
//     auto compute_ctx = func.getArgument(0);

//     auto build_run_ctx = DeclareBuildRunContext(rewriter, &module);
//     rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, build_run_ctx, ValueRange{reg_ctx, compute_ctx});
//     return success();
//   }
// };

// // create okl.kernel(*reg_ctx, StringAttr: op_type_name)
// struct KernelOpLowering final : public OpConversionPattern<KernelOp> {
//   // raw: create okl.kernel(*reg_ctx, StringAttr: op_type_name) -> llvm_ptr<i8>
//   // dst: llvm.call build_kernel(reg_ctx: llvm_ptr<i8>, op_type_name: llvm_ptr<i8>) -> llvm_ptr<i8>
//   static LLVM::LLVMFuncOp DeclareBuildKernel(::mlir::PatternRewriter& rewriter, ModuleOp* module) {
//     auto func_name = "build_kernel";
//     LLVM::LLVMFuncOp func;
//     if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(func_name))) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(module->getBody());

//       auto func_type = LLVM::LLVMFunctionType::get(
//           GetPtrType(rewriter), {GetPtrType(rewriter), GetPtrType(rewriter)}, false);
//       func = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name, func_type,
//                                                LLVM::Linkage::External);
//       func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
//     }
//     return func;
//   }

//  public:
//   using OpConversionPattern<KernelOp>::OpConversionPattern;
//   LogicalResult matchAndRewrite(KernelOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter& rewriter) const override {
//     auto module = GetModuleOpFromJobBodyOp<func::FuncOp>(op);
//     auto build_launch = DeclareBuildKernel(rewriter, &module);

//     auto op_type_name = op.op_type_name();
//     auto global_str = DeclareOrGetGlobalString(rewriter, &module, op_type_name, op_type_name);
//     auto gep = GetGepOpFromGlobal(rewriter, &module, &global_str);
//     auto reg_ctx = op.reg_ctx();

//     rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, build_launch, ValueRange{reg_ctx, gep})
//         ->getResult(0);
//     return success();
//   }
// };

// // create okl.launch(*reg_ctx, *run_ctx, *kernel)
// struct LaunchOpLowering final : public OpConversionPattern<LaunchOp> {
//   // raw: create okl.launch(*reg_ctx, *run_ctx, *kernel) -> llvm_ptr<i8>
//   // dst: llvm.call launch(reg_ctx: llvm_ptr<i8>, run_ctx: llvm_ptr<i8>, kernel: llvm_ptr<i8>)
//   static LLVM::LLVMFuncOp DeclareBuildLaunch(::mlir::PatternRewriter& rewriter, ModuleOp* module) {
//     auto func_name = "launch";
//     LLVM::LLVMFuncOp func;
//     if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(func_name))) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(module->getBody());

//       auto void_type = LLVM::LLVMVoidType::get(rewriter.getContext());
//       auto func_type = LLVM::LLVMFunctionType::get(
//           void_type, {GetPtrType(rewriter), GetPtrType(rewriter), GetPtrType(rewriter)}, false);
//       func = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name, func_type,
//                                                LLVM::Linkage::External);
//       func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
//     }
//     return func;
//   }

//  public:
//   using OpConversionPattern<LaunchOp>::OpConversionPattern;
//   LogicalResult matchAndRewrite(LaunchOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter& rewriter) const override {
//     auto module = GetModuleOpFromJobBodyOp<func::FuncOp>(op);

//     auto build_launch = DeclareBuildLaunch(rewriter, &module);
//     auto reg_ctx = op.reg_ctx();
//     auto run_ctx = op.run_ctx();
//     auto kernel = op.kernel();

//     rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, build_launch,
//                                               ValueRange{reg_ctx, run_ctx, kernel});
//     return success();
//   }
// };

// struct DestroyRegContextOpLowering final : public OpConversionPattern<DestroyRegContextOp> {
//   static LLVM::LLVMFuncOp DeclareDestroyRegContext(::mlir::PatternRewriter& rewriter,
//                                                    ModuleOp* module) {
//     auto func_name = "destroy_reg_ctx";
//     LLVM::LLVMFuncOp func;
//     if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(func_name))) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(module->getBody());

//       auto void_type = LLVM::LLVMVoidType::get(rewriter.getContext());
//       auto func_type = LLVM::LLVMFunctionType::get(void_type, {GetPtrType(rewriter)}, false);
//       func = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name, func_type,
//                                                LLVM::Linkage::External);
//       func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
//     }
//     return func;
//   }

//  public:
//   using OpConversionPattern<DestroyRegContextOp>::OpConversionPattern;
//   LogicalResult matchAndRewrite(DestroyRegContextOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter& rewriter) const override {
//     auto module = GetModuleOpFromJobBodyOp<func::FuncOp>(op);

//     auto build_launch = DeclareDestroyRegContext(rewriter, &module);
//     auto reg_ctx = op.ptr();

//     rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, build_launch, ValueRange{reg_ctx});
//     return success();
//   }
// };

// struct DestroyRunContextOpLowering final : public OpConversionPattern<DestroyRunContextOp> {
//   static LLVM::LLVMFuncOp DeclareDestroyRunContext(::mlir::PatternRewriter& rewriter,
//                                                    ModuleOp* module) {
//     auto func_name = "destroy_reg_ctx";
//     LLVM::LLVMFuncOp func;
//     if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(func_name))) {
//       OpBuilder::InsertionGuard guard(rewriter);
//       rewriter.setInsertionPointToStart(module->getBody());

//       auto void_type = LLVM::LLVMVoidType::get(rewriter.getContext());
//       auto func_type = LLVM::LLVMFunctionType::get(void_type, {GetPtrType(rewriter)}, false);
//       func = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), func_name, func_type,
//                                                LLVM::Linkage::External);
//       func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
//     }
//     return func;
//   }

//  public:
//   using OpConversionPattern<DestroyRunContextOp>::OpConversionPattern;
//   LogicalResult matchAndRewrite(DestroyRunContextOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter& rewriter) const override {
//     auto module = GetModuleOpFromJobBodyOp<func::FuncOp>(op);

//     auto build_launch = DeclareDestroyRunContext(rewriter, &module);
//     auto reg_ctx = op.ptr();

//     rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, build_launch, ValueRange{reg_ctx});
//     return success();
//   }
// };
namespace {
struct LowerOKLToLLVMPass : public LowerOKLToLLVMPassBase<LowerOKLToLLVMPass> {
  void runOnOperation() override;
};
}  // namespace

std::unique_ptr<Pass> createLowerOKLToLLVMPass() { return std::make_unique<LowerOKLToLLVMPass>(); }

void LowerOKLToLLVMPass::runOnOperation() {
  MLIRContext* context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<oneflow::OneFlowDialect, LLVM::LLVMDialect, func::FuncDialect>();
  target.addIllegalDialect<okl::OKLDialect>();

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  RewritePatternSet patterns(context);

  // patterns.add<KernelOpLowering, LaunchOpLowering, RegContextOpLowering, RunContextOpLowering,
  //              DestroyRegContextOpLowering, DestroyRunContextOpLowering>(typeConverter, context);
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
    getOperation()->emitError("Failed to lower OKL to LLVM");
  }
}

}  // namespace okl

}  // namespace mlir
