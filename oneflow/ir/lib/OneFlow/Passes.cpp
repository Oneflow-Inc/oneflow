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
#include "OneFlow/OneFlowOps.h"
#include <iostream>
#include <string>
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::oneflow;

::llvm::SmallVector<::mlir::Value, 4> OutlineFunction(::mlir::PatternRewriter& rewriter,
                                                      mlir::OpResult mul_res,
                                                      mlir::OpResult cast_res) {
  // get matched scale and cast
  // create JIT op and kernel

  if (auto mul_op = llvm::dyn_cast<ScalarMulByTensorOp>(mul_res.getDefiningOp())) {
    if (auto cast_op = llvm::dyn_cast<CastOp>(cast_res.getDefiningOp())) {
      NamedAttrList attributes;
      attributes.set("op_type_name", rewriter.getStringAttr("mlir_jit"));
      // TODO: extract a function to strip attrs from an op to be replace
      attributes.set("device_tag", mul_op.device_tagAttr());
      attributes.set("device_name", mul_op.device_nameAttr());
      attributes.set("hierarchy", mul_op.hierarchyAttr());
      using LBNVec = SmallVector<StringRef, 8>;
      using LBNSegVec = SmallVector<int32_t, 8>;

      LBNVec input_lbn_segment_keys;
      LBNSegVec input_lbn_segment_sizes;
      input_lbn_segment_keys.push_back("in");
      input_lbn_segment_sizes.push_back(1);

      attributes.set("input_lbn_segment_keys", rewriter.getStrArrayAttr(input_lbn_segment_keys));
      attributes.set("input_lbn_segment_sizes", rewriter.getI32ArrayAttr(input_lbn_segment_sizes));

      // TODO: extract a function to generate op name for jit op from ops being fused
      SmallString<64> op_name_storage;
      auto op_name =
          (cast_op.op_name() + "__FUSE__" + mul_op.op_name()).toStringRef(op_name_storage);
      attributes.set("op_name", rewriter.getStringAttr(op_name));

      LBNVec output_lbns;
      LBNVec output_lbn_segment_keys;
      LBNSegVec output_lbn_segment_sizes;
      // TODO: use functions in oneflow to genearated bn
      SmallString<64> output_lbn_storage;
      output_lbns.push_back((op_name + "/" + "out_0").toStringRef(output_lbn_storage));
      output_lbn_segment_keys.push_back("out");
      output_lbn_segment_sizes.push_back(1);
      attributes.set("output_lbns", rewriter.getStrArrayAttr(output_lbns));
      attributes.set("output_lbn_segment_keys", rewriter.getStrArrayAttr(output_lbn_segment_keys));
      attributes.set("output_lbn_segment_sizes",
                     rewriter.getI32ArrayAttr(output_lbn_segment_sizes));

      attributes.set("scope_symbol_id", mul_op.scope_symbol_idAttr());
      SmallVector<::mlir::Value, 2> operands;
      operands.push_back(cast_op.x());
      operands.push_back(mul_op.scalar());
      auto created = rewriter.create<MlirJitOp>(mul_op.getLoc(),
                                                /* resultTypes */ mul_op->getResultTypes(),
                                                /* operands */ operands,
                                                /* attributes */ attributes);
      cast_op.replaceAllUsesWith(created);

      // TODO: is it a good idea to insert the sub-graph at entry block?
      // TODO: add dedicated op definition for this kind OneFlow_JitFunc
      // TODO: save input output alias info in OneFlow_JitFunc's attr
      auto* context = rewriter.getContext();
      OpBuilder builder(context);

      OwningModuleRef jit_module(
          ModuleOp::create(FileLineColLoc::get(context, "", /*line=*/0, /*column=*/0)));

      // create a function to be lowered
      // TODO: erase tensor shapes in the outlined function
      // TODO: use infer shape interface to infer shapes at different passes and runtime
      auto func_type =
          rewriter.getFunctionType(created->getOperandTypes(), created->getResultTypes());
      auto function = builder.create<mlir::FuncOp>(mul_op->getLoc(), op_name, func_type);

      auto& entry_block = *function.addEntryBlock();
      builder.setInsertionPointToStart(&entry_block);

      // TODO: make this transformation generic, using a value => arg mapping and walk the graph
      auto cast_op_ =
          builder.create<CastOp>(cast_op->getLoc(), /* resultTypes */ cast_op->getResultTypes(),
                                 /* operands */ entry_block.getArguments().take_front(),
                                 /* attributes */ cast_op->getAttrs());
      auto scalar_mul = builder.create<ScalarMulByTensorOp>(
          mul_op->getLoc(), /* resultTypes */ mul_op->getResultTypes(),
          /* operands */
          SmallVector<::mlir::Value, 2>({cast_op_.y(), entry_block.getArgument(1)}),
          /* attributes */ mul_op->getAttrs());
      builder.create<ReturnOp>(mul_op->getLoc(), scalar_mul.y());
      jit_module->push_back(function);
      std::string mlir;
      llvm::raw_string_ostream os_mlir(mlir);
      jit_module->print(os_mlir);
      created->setAttr("mlir_assembly", rewriter.getStringAttr(mlir));
      cast_op.erase();
      return created->getResults();
    }
  }
  // TODO: raise a more reasonable error
  return {};
}

#include "OneFlow/OneFlowPatterns.cpp.inc"

namespace mlir {

namespace oneflow {

LogicalResult Lower(mlir::MLIRContext* context, OwningModuleRef& module) {
  mlir::PassManager pm(context);
  pm.addPass(createLowerOneFlowToTosaPass());                     // -lower-oneflow-to-tosa
  pm.addPass(createCSEPass());                                    // -cse
  pm.addNestedPass<FuncOp>(tosa::createTosaToLinalgOnTensors());  // -tosa-to-linalg-on-tensors
  pm.addNestedPass<FuncOp>(createLinalgFusionOfTensorOpsPass());  // -linalg-fusion-for-tensor-ops
  pm.addNestedPass<FuncOp>(createLinalgBufferizePass());          // -linalg-bufferize
  pm.addNestedPass<FuncOp>(createTensorBufferizePass());          // -tensor-bufferize
  pm.addPass(createTensorConstantBufferizePass());                // --tensor-constant-bufferize
  pm.addPass(createFuncBufferizePass());                          // -func-bufferize
  pm.addPass(createBufferResultsToOutParamsPass());               // -buffer-results-to-out-params
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());     // -convert-linalg-to-loops
  pm.addNestedPass<FuncOp>(createLowerToCFGPass());               // -convert-scf-to-std
  pm.addPass(createConvertLinalgToLLVMPass());                    // -convert-linalg-to-llvm
  pm.addPass(createLowerToLLVMPass());                            // -convert-std-to-llvm
  pm.addPass(createMemRefToLLVMPass());                           // -convert-memref-to-llvm
  return pm.run(module.get());
}

void populateFuserPasses(::mlir::RewritePatternSet& patterns) {
  patterns.add<OutlineFuseCastScale>(patterns.getContext());
}

}  // namespace oneflow

}  // namespace mlir
