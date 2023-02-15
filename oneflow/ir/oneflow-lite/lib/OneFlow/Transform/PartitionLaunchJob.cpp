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
#include "OneFlow/Transform/PartitionLaunchJob.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace oneflow {
namespace lite {

struct PartitionLaunchJobPass
    : public PassWrapper<PartitionLaunchJobPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;

  bool needPartition(StringRef device) const { return device == "tensorrt" || device == "ascend"; }

  func::FuncOp addCallableFunc(OpBuilder& builder, StringRef callee_name,
                               const llvm::SmallVector<Value, 4>& operands,
                               const llvm::SmallVector<Value, 4>& results,
                               const llvm::SmallVector<Operation*, 4>& block);
};

func::FuncOp PartitionLaunchJobPass::addCallableFunc(
    OpBuilder& builder, StringRef callee_name, const llvm::SmallVector<Value, 4>& operands,
    const llvm::SmallVector<Value, 4>& results, const llvm::SmallVector<Operation*, 4>& block) {
  llvm::SmallVector<Type, 4> operand_types, result_types;
  for (auto operand : operands) { operand_types.push_back(operand.getType()); }
  for (auto result : results) { result_types.push_back(result.getType()); }

  auto parentFuncOp = block[0]->getParentOfType<oneflow::Job>();
  auto parentModuleOp = parentFuncOp->getParentOfType<ModuleOp>();

  Block::iterator insertPt(parentFuncOp->getNextNode());
  builder.setInsertionPointToStart(parentModuleOp.getBody());

  auto funcType = builder.getFunctionType(operand_types, result_types);
  auto funcOp = builder.create<func::FuncOp>(block[0]->getLoc(), callee_name, funcType);
  auto* entryBlock = funcOp.addEntryBlock();

  BlockAndValueMapping mapping;
  for (auto operand : llvm::enumerate(operands)) {
    mapping.map(operand.value(), entryBlock->getArgument(operand.index()));
  }

  builder.setInsertionPointToStart(entryBlock);
  for (Operation* op : block) {
    builder.insert(op->clone(mapping));
    for (auto result : llvm::enumerate(op->getResults())) {
      mapping.map(result.value(), entryBlock->back().getResult(result.index()));
    }
  }
  llvm::SmallVector<Value, 4> mappingResults;
  for (auto result : results) { mappingResults.push_back(mapping.lookup(result)); }
  builder.create<func::ReturnOp>(block[0]->getLoc(), mappingResults);
  return funcOp;
}

void PartitionLaunchJobPass::runOnOperation() {
  // TODO(): refactor
  llvm::DenseMap<StringRef, llvm::SetVector<Operation*, llvm::SmallVector<Operation*, 4>>>
      partitionOps;
  getOperation().walk([&](Operation* op) {
    if (!op->hasTrait<OpTrait::IsOpConfCompatible>()) { return; }
    if (dyn_cast<CopyOp>(op)) { return; }
    auto device =
        op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr());
    if (!needPartition(device.getValue())) { return; }

    partitionOps[device.getValue()].insert(op);
  });

  for (auto it : partitionOps) {
    if (it.second.empty()) { continue; }

    llvm::DenseMap<Value, int> inputVals, resultVals;
    for (Operation* op : it.second) {
      for (Value operand : op->getOperands()) {
        if (!it.second.count(operand.getDefiningOp())) {
          inputVals.try_emplace(operand, inputVals.size());
        }
      }
      for (Value result : op->getResults()) {
        for (auto& use : result.getUses()) {
          if (!it.second.count(use.getOwner())) {
            resultVals.try_emplace(result, resultVals.size());
            break;
          }
        }
      }
    }
    auto block = it.second.takeVector();
    // TODO(): check job is acyclic or not
    llvm::SmallVector<Value, 4> operands(inputVals.size());
    llvm::SmallVector<Value, 4> results(resultVals.size());
    for (auto in : inputVals) { operands[in.second] = in.first; }
    for (auto out : resultVals) { results[out.second] = out.first; }

    OpBuilder builder(&getContext());
    auto callableFunc =
        addCallableFunc(builder, it.first.str() + ".launch", operands, results, block);

    Operation* firstOp = block[0];
    NamedAttrList attributes;
    attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(),
                   OpTrait::IsOpConfCompatible<void>::getDeviceTag(firstOp));
    attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr(),
                   OpTrait::IsOpConfCompatible<void>::getDeviceName(firstOp));
    if (auto hierarchy = OpTrait::IsOpConfCompatible<void>::getHierarchy(firstOp)) {
      attributes.set(OpTrait::IsOpConfCompatible<void>::getHierarchyAttr(), hierarchy);
    }
    attributes.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                   builder.getStringAttr(it.first.str() + ".launch"));
    if (auto scope_symbol_id = OpTrait::IsOpConfCompatible<void>::getScopeSymbolID((firstOp))) {
      attributes.set(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr(), scope_symbol_id);
    }
    builder.setInsertionPointAfter(firstOp);

    auto launchOp =
        builder.create<MlirJitOp>(firstOp->getLoc(), callableFunc, attributes, operands);
    launchOp->setAttr("mlir_assembly", builder.getStringAttr(""));

    for (auto result : llvm::enumerate(results)) {
      result.value().replaceAllUsesWith(launchOp->getOperand(result.index()));
    }
    for (Operation* op : block) {
      op->dropAllUses();
      op->erase();
    }
  }
}

std::unique_ptr<mlir::Pass> createLitePartitionLaunchJobPass() {
  return std::unique_ptr<mlir::Pass>(new PartitionLaunchJobPass());
}

}  // namespace lite
}  // namespace oneflow
}  // namespace mlir
