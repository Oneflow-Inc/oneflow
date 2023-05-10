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
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowUtils.h"
#include "OneFlow/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Bytecode/BytecodeWriter.h"

namespace mlir {

namespace oneflow {

namespace {

// general lowering path:
// 1. outline linalg ops to a func.func and an oneflow.jit op
// 2. bufferize the func.func and update oneflow.jit op's tmp buffer size

// 1. collect ops to outline
// 2. create func.func jit ops to call
// 3. replace the usages with jit ops' results

// entries: non-oneflow ops which have operands are from oneflow ops
// exits: result consumed by oneflow ops

// NOTE: we assume all arg values are produced by an oneflow op and won't be an argument

NamedAttrList GetJitOpAttributes(Builder& rewriter, StringRef op_name, int32_t input_size,
                                 int32_t output_size, Operation* op) {
  NamedAttrList attributes;
  attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(),
                 OpTrait::IsOpConfCompatible<void>::getDeviceTag(op));
  attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr(),
                 OpTrait::IsOpConfCompatible<void>::getDeviceName(op));
  if (auto hierarchy = OpTrait::IsOpConfCompatible<void>::getHierarchy(op)) {
    attributes.set(OpTrait::IsOpConfCompatible<void>::getHierarchyAttr(), hierarchy);
  }
  attributes.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                 rewriter.getStringAttr(op_name));
  if (auto scope_symbol_id = OpTrait::IsOpConfCompatible<void>::getScopeSymbolID(op)) {
    attributes.set(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr(), scope_symbol_id);
  }
  return attributes;
}

bool isOneFlowOp(Operation* op) { return llvm::dyn_cast<OneFlowDialect>(op->getDialect()); }
class Outliner {
 private:
  OpBuilder& builder;
  Block* body;
  llvm::DenseSet<Operation*>& visitedOps;
  std::queue<Operation*> worklist{};
  void cloneOpsToNewBody(Operation* op, bool defer = false) {
    if (visitedOps.contains(op)) { return; }
    for (auto operand : op->getOperands()) {
      if (!mapping.lookupOrNull(operand)) {
        if (auto defOp = operand.getDefiningOp()) {
          if (isOneFlowOp(defOp)) {
            entries.insert(operand);
            auto arg = body->addArgument(operand.getType(), operand.getLoc());
            mapping.map(operand, arg);
            mappingReversed.map(arg, operand);
          } else {
            cloneOpsToNewBody(defOp, true);
          }
        }
      }
    }
    ImplicitLocOpBuilder nb(op->getLoc(), builder);
    nb.clone(*op, mapping);
    visitedOps.insert(op);

    for (auto& use : op->getUses()) {
      auto owner = use.getOwner();
      if (isOneFlowOp(owner)) {
        exits.insert(use.get());
      } else {
        if (defer) {
          worklist.push(owner);
        } else {
          cloneOpsToNewBody(owner);
        }
      }
    }
    if (!defer) {
      while (!worklist.empty()) {
        auto op = worklist.front();
        worklist.pop();
        cloneOpsToNewBody(op);
      }
    }
  }

 public:
  Outliner(OpBuilder& builder, Block* body, Operation* op, llvm::DenseSet<Operation*>& visitedOps)
      : builder{builder}, body{body}, visitedOps{visitedOps} {
    cloneOpsToNewBody(op);
  }

  IRMapping mapping{};
  IRMapping mappingReversed{};
  llvm::DenseSet<Value> entries{}, exits{};
};

static std::string JITOpNamePrefix = "JITOpGenerated";
int64_t getCountJITFunction() {
  static std::atomic_int64_t countJITFunction = 0;
  return countJITFunction.fetch_add(1);
}

namespace {

std::function<void(mlir::MLIRContext* mlir_ctx, mlir::ModuleOp module)> getLowerFunction(
    const StringAttr& device_tag) {
  auto device_tag_str = device_tag.str();
#ifdef WITH_MLIR_CUDA_CODEGEN
  if (device_tag_str == "cuda") {
    return [](mlir::MLIRContext* mlir_ctx, mlir::ModuleOp module) {
      CHECK(mlir::succeeded(mlir::oneflow::LowerModuleToCUDALLVM(mlir_ctx, module)))
          << "fail to lower OneFlow to CUDA LLVM";
    };
  }
#endif  // WITH_MLIR_CUDA_CODEGEN
  if (device_tag_str == "cpu") {
    return [](mlir::MLIRContext* mlir_ctx, mlir::ModuleOp module) {
      CHECK(mlir::succeeded(mlir::oneflow::LowerModuleToLLVM(mlir_ctx, module)))
          << "fail to lower OneFlow to LLVM";
    };
  }
  LOG(FATAL) << "Fail to match lowering function with device tag name: " << device_tag_str;
}
std::string convertFuncToByte(func::FuncOp& func) {
  std::string byte;
  llvm::raw_string_ostream os_byte(byte);
  mlir::writeBytecodeToFile(func, os_byte);
  return byte;
}

std::string lowerFuncToLLVMByte(const std::string& raw_byte, const StringAttr& device_tag) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::oneflow::OneFlowDialect>();
  mlir::MLIRContext mlir_ctx(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      ::mlir::parseSourceString<mlir::ModuleOp>(raw_byte, &mlir_ctx);
  mlir::registerLLVMDialectTranslation(registry);
  if (::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_STDOUT", false)) { module->print(llvm::outs()); }
  getLowerFunction(device_tag)(&mlir_ctx, *module);
  if (::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_STDOUT", false)) { module->print(llvm::outs()); }
  (*module)->setAttr(jit::RAW_GRAPH, StringAttr::get(&mlir_ctx, raw_byte));

  std::string byte;
  llvm::raw_string_ostream os_byte(byte);
  mlir::writeBytecodeToFile(*module, os_byte);
  return byte;
}

}  // namespace

class OutlineJitFunctionPass : public OutlineJitFunctionPassBase<OutlineJitFunctionPass> {
  void runOnOperation() override {
    llvm::DenseSet<Operation*> entryOps, visitedOps;
    FunctionOpInterface job = getOperation();
    auto& operations = job.getFunctionBody().front().getOperations();

    for (auto& op : operations) {
      if (llvm::dyn_cast<OneFlowDialect>(op.getDialect())) {
        for (auto result : op.getResults()) {
          for (auto user : result.getUsers()) {
            if (!isOneFlowOp(user)) { entryOps.insert(user); }
          }
        }
      }
    }

    OpBuilder builder{&getContext()};
    for (auto entryOp : entryOps) {
      if (visitedOps.contains(entryOp)) { continue; }
      OpBuilder::InsertionGuard guard(builder);
      auto block = new Block();
      builder.setInsertionPointToStart(block);
      auto outliner = Outliner(builder, block, entryOp, visitedOps);

      SmallVector<::mlir::Value, 4> entries, exits, mappedExits;
      SmallVector<Type, 4> argumentTypes, resultTypes;

      for (Value exit : outliner.exits) {
        exits.push_back(exit);
        mappedExits.push_back(outliner.mapping.lookup(exit));
        resultTypes.push_back(exit.getType());
      }
      builder.setInsertionPointToEnd(block);
      builder.create<func::ReturnOp>(entryOp->getLoc(), mappedExits);

      for (auto argument : block->getArguments()) {
        if (auto found = outliner.mappingReversed.lookupOrNull(argument)) {
          entries.push_back(found);
          argumentTypes.push_back(argument.getType());
        } else {
          job->emitError() << "fail to outline, entry not found for argument #"
                           << argument.getArgNumber();
          signalPassFailure();
        }
      }
      auto funcType = builder.getFunctionType(argumentTypes, resultTypes);
      if (auto mod = job->getParentOfType<ModuleOp>()) {
        auto name = JITOpNamePrefix + std::to_string(getCountJITFunction());
        SmallString<16> tempBuffer;
        name = SanitizeIdentifier(name, tempBuffer);

        builder.setInsertionPointToStart(&mod.getRegion().front());
        auto function = builder.create<func::FuncOp>(entryOp->getLoc(), name, funcType);
        function.getBody().push_front(block);

        if (auto lastOp = exits.back().getDefiningOp()) {
          builder.setInsertionPointAfter(lastOp);
          NamedAttrList attributes =
              GetJitOpAttributes(builder, name, argumentTypes.size(), resultTypes.size(),
                                 entryOp->getOperand(0).getDefiningOp());
          std::string byte =
              compileToLLVM.getValue() ? lowerFuncToLLVMByte(
                  convertFuncToByte(function),
                  attributes.get(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr())
                      .cast<StringAttr>())
                                       : convertFuncToByte(function);
          auto jitOp = builder.create<MlirJitOp>(entryOp->getLoc(), function, attributes, entries);
          jitOp->setAttr("mlir_assembly", builder.getStringAttr(byte));
          for (const auto& old : llvm::enumerate(exits)) {
            old.value().replaceAllUsesWith(jitOp->getResult(old.index()));
          }
        } else {
          job->emitError() << "fail to outline, nowhere to replace";
          signalPassFailure();
        }
      } else {
        job->emitError() << "fail to outline";
        signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createOutlineJitFunctionPass() {
  return std::make_unique<OutlineJitFunctionPass>();
}

}  // namespace oneflow

}  // namespace mlir
