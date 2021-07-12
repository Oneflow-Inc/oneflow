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
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::oneflow;

static mlir::ParseResult parseConstantOp(mlir::OpAsmParser& parser, mlir::OperationState& result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes)
      || parser.parseAttribute(value, "value", result.attributes)) {
    return failure();
  }
  result.addTypes(value.getType());
  return success();
}

static mlir::LogicalResult verify(oneflow::ConstantOp op) { return mlir::success(); }

template<typename OpType>
LogicalResult TrimRedundantCtrl(OpType& op, PatternRewriter& rewriter) {
  if (op.ctrl_output() && op.ctrl_output().use_empty()) {
    const int32_t num_data_outputs =
        *(op.result_segment_sizes().template getValues<uint32_t>()).begin();
    NamedAttrList attributes(op->getAttrDictionary());
    attributes.erase("result_segment_sizes");
    attributes.append("result_segment_sizes", rewriter.getI32VectorAttr({num_data_outputs, 0}));
    if (auto created =
            rewriter.create<OpType>(op->getLoc(), op.getODSResults(0 /* data out */).getTypes(),
                                    op->getOperands(), attributes)) {
      for (auto out : op.data_output()) {
        out.replaceAllUsesWith(created->getResult(out.getResultNumber()));
      }
      op->erase();
      return success();
    }
  }
  return failure();
}

bool IsCtrlOutTrimmed(oneflow::UserOp& op) { return !op.ctrl_output(); }

bool IsCtrlInAbsent(oneflow::UserOp& op) { return op.ctrl_inputs().empty(); }

struct ConcreteUserOps : public mlir::OpRewritePattern<oneflow::UserOp> {
  ConcreteUserOps(mlir::MLIRContext* context)
      : OpRewritePattern<oneflow::UserOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(oneflow::UserOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    auto op_type_name = op->getAttrOfType<StringAttr>("op_type_name").getValue();
    op.getODSResults(0);
    if (succeeded(TrimRedundantCtrl(op, rewriter))) {
      return success();
    }
    // In principle, a concrete user op has no ctrl input/output. Some benefits:
    // 1. simplify things
    // 2. make conversion and code gen more doable
    // 3. enable the reuse of established MLIR infra like built-in traits
    else if (IsCtrlOutTrimmed(op) && IsCtrlInAbsent(op)) {
      if (/* convert opaque elementwise user op to a concrete op */ op_type_name.equals("abs")
          || op_type_name.equals("ceil") || op_type_name.equals("floor")
          || op_type_name.equals("relu") || op_type_name.equals("rint")
          || op_type_name.equals("round") || op_type_name.equals("sign")
          || op_type_name.equals("negative") || op_type_name.equals("reciprocal")
          || op_type_name.equals("cast")) {
        NamedAttrList attributes(op->getAttrDictionary());
        attributes.erase("operand_segment_sizes");
        attributes.erase("result_segment_sizes");
        auto unknownLoc = UnknownLoc::get(rewriter.getContext());
        OperationState state(unknownLoc, "oneflow." + op_type_name.str());
        state.addAttributes(attributes);
        state.addOperands(op->getOperands());
        assert(op.data_input().size() == 1);
        assert(op.data_output().size() == 1);
        state.addTypes(op.getODSResults(0 /* data out */).getTypes());
        if (auto elementwise = rewriter.createOperation(state)) {
          op.data_output().front().replaceAllUsesWith(elementwise->getResult(0));
          op->erase();
          return success();
        }
      } else if (op_type_name.equals("scalar_mul_by_tensor")) {
        assert(op.data_input().size() == 2);
        assert(op.data_output().size() == 1);
        // TODO: refine repetitive code
        NamedAttrList attributes(op->getAttrDictionary());
        attributes.erase("operand_segment_sizes");
        attributes.erase("result_segment_sizes");
        auto unknownLoc = UnknownLoc::get(rewriter.getContext());
        OperationState state(unknownLoc, "oneflow." + op_type_name.str());
        state.addAttributes(attributes);
        SmallVector<::mlir::Value, 2> operands;
        for (std::tuple<const mlir::Attribute&, mlir::Value> kv :
             llvm::zip(op.input_lbn_segment_keysAttr(), op.data_input())) {
          auto k = std::get<0>(kv).dyn_cast<StringAttr>().getValue();
          auto v = std::get<1>(kv);
          if (k.equals("x")) { operands.insert(operands.begin(), v); }
          if (k.equals("scalar")) { operands.insert(operands.end(), v); }
        }
        state.addOperands(operands);
        state.addTypes(op.getODSResults(0 /* data out */).getTypes());
        if (auto elementwise = rewriter.createOperation(state)) {
          op.data_output().front().replaceAllUsesWith(elementwise->getResult(0));
          op->erase();
          return success();
        }
      }
    }

    return failure();
  }
};

void UserOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                         ::mlir::MLIRContext* context) {
  results.insert<ConcreteUserOps>(context);
}

struct ConcreteSystemOps : public mlir::OpRewritePattern<oneflow::SystemOp> {
  ConcreteSystemOps(mlir::MLIRContext* context)
      : OpRewritePattern<oneflow::SystemOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(oneflow::SystemOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    return TrimRedundantCtrl(op, rewriter);
  }
};

void SystemOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                           ::mlir::MLIRContext* context) {
  results.insert<ConcreteSystemOps>(context);
}

// TODO: merge all ctrl input and output when folding op
bool HaveIdenticalPlacement(mlir::Operation* a, mlir::Operation* b) {
  UserOpAdaptor adaptor_a(a->getOperands(), a->getAttrDictionary());
  UserOpAdaptor adaptor_b(b->getOperands(), b->getAttrDictionary());
  return adaptor_a.device_tag() == adaptor_b.device_tag()
         && adaptor_a.device_name() == adaptor_b.device_name();
}

OpFoldResult OpTrait::impl::foldIdempotentOfIdenticalPlacement(Operation* op) {
  auto* argument_op = op->getOperand(0).getDefiningOp();
  if (argument_op && op->getName() == argument_op->getName()
      && HaveIdenticalPlacement(op, argument_op)) {
    return op->getOperand(0);
  }
  return {};
}

OpFoldResult OpTrait::impl::foldInvolutionOfIdenticalPlacement(Operation* op) {
  auto* argument_op = op->getOperand(0).getDefiningOp();
  if (argument_op && op->getName() == argument_op->getName()
      && HaveIdenticalPlacement(op, argument_op)) {
    return argument_op->getOperand(0);
  }
  return {};
}

static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

using LoopIterationFn =
    function_ref<Value(OpBuilder& rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static Value insertAllocAndDealloc(MemRefType type, Location loc, PatternRewriter& rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);
  auto* parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

static void lowerOpToLoops(Operation* op, ValueRange operands, PatternRewriter& rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();
  auto memRefType = convertTensorToMemRef(tensorType);
  // TODO: remove alloc and dealloc
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  buildAffineLoopNest(rewriter, loc, lowerBounds, tensorType.getShape(), steps,
                      [&](OpBuilder& nestedBuilder, Location loc, ValueRange ivs) {
                        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
                        nestedBuilder.create<AffineStoreOp>(loc, valueToStore, alloc, ivs);
                      });
  rewriter.replaceOp(op, alloc);
}

struct ScalarMulByTensorOpLowering final : public OpConversionPattern<ScalarMulByTensorOp> {
 public:
  using OpConversionPattern<ScalarMulByTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ScalarMulByTensorOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder& builder, ValueRange memRefOperands, ValueRange loopIvs) {
                     typename ScalarMulByTensorOp::Adaptor adaptor(memRefOperands);
                     auto loadedLhs = builder.create<AffineLoadOp>(loc, adaptor.x(), loopIvs);
                     auto loadedRhs = builder.create<AffineLoadOp>(loc, adaptor.scalar(), loopIvs);
                     return builder.create<MulFOp>(loc, loadedLhs, loadedRhs);
                   });
    return success();
  }
};

struct CastOpLowering final : public OpConversionPattern<CastOp> {
 public:
  using OpConversionPattern<CastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(CastOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    auto loc = op->getLoc();
    typename CastOp::Adaptor adaptor(operands);
    if (!adaptor.x().getType().dyn_cast<MemRefType>()) { return failure(); }
    return failure();
  }
};

class FuncOpConversion final : public OpConversionPattern<FuncOp> {
 public:
  using OpConversionPattern<FuncOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(FuncOp func, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const final {
    auto func_type = func.getType();
    TypeConverter::SignatureConversion conversion(func_type.getNumInputs());
    // TODO: handle input output alias here by adding extra input arg
    for (auto arg_type : llvm::enumerate(func_type.getInputs())) {
      auto converted = convertTensorToMemRef(arg_type.value().cast<TensorType>());
      conversion.addInputs(arg_type.index(), converted);
    }

    rewriter.applySignatureConversion(&func.getBody(), conversion);
    rewriter.updateRootInPlace(func, [&] {
      func.setType(
          rewriter.getFunctionType(conversion.getConvertedTypes(), func_type.getResults()));
    });
    return success();
  }
};

namespace {
struct AffineLoweringPass : public LowerOneFlowToAffinePassBase<AffineLoweringPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, StandardOpsDialect>();
  }
  StringRef getName() const override { return "AffineLoweringPass"; }
  void runOnOperation() override;
};
}  // namespace

std::unique_ptr<Pass> mlir::oneflow::createLowerOneFlowToAffinePass() {
  return std::make_unique<AffineLoweringPass>();
}

void AffineLoweringPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, memref::MemRefDialect, StandardOpsDialect>();
  target.addIllegalDialect<OneFlowDialect>();
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    for (auto arg : op.getArguments()) {
      if (arg.getType().dyn_cast<TensorType>()) { return false; }
    }
    return true;
  });
  RewritePatternSet patterns(&getContext());
  // TODO: Add type converter
  patterns.insert<CastOpLowering, ScalarMulByTensorOpLowering, FuncOpConversion>(&getContext());
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    getOperation()->dump();
    signalPassFailure();
  }
}

LogicalResult Lower(mlir::MLIRContext& context, OwningModuleRef& module) {
  context.getOrLoadDialect<oneflow::OneFlowDialect>();
  context.loadDialect<StandardOpsDialect>();
  context.loadDialect<memref::MemRefDialect>();

  mlir::PassManager pm(&context);
  pm.addPass(createLowerOneFlowToAffinePass());
  pm.dump();
  return pm.run(module.get());
}

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

      mlir::MLIRContext context;

      // TODO: is it a good idea to insert the sub-graph at entry block?
      // TODO: add dedicated op definition for this kind OneFlow_JitFunc
      // TODO: save input output alias info in OneFlow_JitFunc's attr
      OpBuilder builder(&context);

      OwningModuleRef jit_module(
          ModuleOp::create(FileLineColLoc::get(&context, "", /*line=*/0, /*column=*/0)));

      // create a function to be lowered
      SmallVector<Type, 3> types(created->getOperandTypes());
      types.push_back(created->getResultTypes().front());
      auto func_type = rewriter.getFunctionType(types, llvm::None);
      auto function = builder.create<mlir::FuncOp>(mul_op->getLoc(), op_name, func_type);

      auto& entry_block = *function.addEntryBlock();
      builder.setInsertionPointToStart(&entry_block);

      // TODO: make this transformation generic, using a value => arg mapping and walk the graph
      auto cast_op_ =
          builder.create<CastOp>(cast_op->getLoc(), /* resultTypes */ cast_op->getResultTypes(),
                                 /* operands */ entry_block.getArguments().take_front(),
                                 /* attributes */ cast_op->getAttrs());
      builder.create<ScalarMulByTensorOp>(
          mul_op->getLoc(), /* resultTypes */ mul_op->getResultTypes(),
          /* operands */
          SmallVector<::mlir::Value, 2>({cast_op_.y(), entry_block.getArgument(1)}),
          /* attributes */ mul_op->getAttrs());
      builder.create<ReturnOp>(mul_op->getLoc());
      // TODO: decare terminator
      // TODO: skip outline functions when translating beck to job
      jit_module->push_back(function);
      jit_module->dump();

      LogicalResult result = Lower(context, jit_module);
      if (result.failed()) { exit(EXIT_FAILURE); }
      jit_module->dump();

      cast_op.erase();
      return created->getResults();
    }
  }
  // TODO: raise a more reasonable error
  return {};
}

#include "OneFlow/OneFlowEnums.cpp.inc"
#include "OneFlow/OneFlowPatterns.cpp.inc"

void populateFuserPasses(::mlir::RewritePatternSet& patterns) {
  patterns.add<OutlineFuseCastScale>(patterns.getContext());
}

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.cpp.inc"
