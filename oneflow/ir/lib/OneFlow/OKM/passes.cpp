#include "OneFlow/OKM/OKMDialect.h"
#include "OneFlow/OKM/OKMOps.h"
#include "OneFlow/OKM/passes.h"
#include "OneFlow/OneFlowDialect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace okm {
namespace func_name {

const auto* TAG_NAME = "compiled";
const auto* GRAPH_NAME = "subgraph";
const auto* MEM_GRAPH_NAME = "mem_subgraph";
const auto* WRAP_GRAPH_NAME = "mem_wrap_subgraph";
const auto* OPT_GRAPH_NAME = "mem_opt_subgraph";
}  // namespace func_name
struct ExtractOKMTensorPattern : public mlir::OpRewritePattern<func::FuncOp> {
  static void ExtractArgTensors(func::FuncOp op, mlir::PatternRewriter& rewriter) {
    auto& body = op.getBody();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&body.front());

    for (const auto& arg : llvm::enumerate(op.getBody().getArguments())) {
      auto tensor =
          rewriter.create<okm::ArgToTensorOp>(op->getLoc(), arg.value().getType(), arg.index());
      arg.value().replaceAllUsesWith(tensor);
    }
  }

  static void ExtractRetTensors(func::FuncOp op, mlir::PatternRewriter& rewriter) {
    auto& return_op = op.getBody().front().back();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&return_op);

    llvm::SmallVector<Value> returns;
    for (const auto& ret_val : llvm::enumerate(return_op.getOperands())) {
      auto new_ret = rewriter.create<okm::TensorToRetOp>(op->getLoc(), ret_val.value().getType(),
                                                         ret_val.value(), ret_val.index());
      returns.push_back(new_ret);
    }

    rewriter.replaceOpWithNewOp<func::ReturnOp>(&return_op, ValueRange{returns});
  }

  explicit ExtractOKMTensorPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    const auto sym_name = op.getSymName();
    if (sym_name.startswith(func_name::GRAPH_NAME)) {
      // rename function
      const auto index = sym_name.substr(strlen(func_name::GRAPH_NAME));
      const auto rename = func_name::MEM_GRAPH_NAME + index;
      op.setSymNameAttr(rewriter.getStringAttr(rename));
      // extract tensors
      ExtractArgTensors(op, rewriter);
      ExtractRetTensors(op, rewriter);
      return success();
    }
    return failure();
  }
};

class ExtractOKMTensorPass : public ExtractOKMTensorPassBase<ExtractOKMTensorPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<oneflow::OneFlowDialect>();
    registry.insert<OKMDialect>();
  }

  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    patterns.add<ExtractOKMTensorPattern>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

std::unique_ptr<Pass> createExtractOKMTensorPass() {
  return std::make_unique<ExtractOKMTensorPass>();
}

struct WrapOKMKernelPattern : public mlir::OpRewritePattern<func::FuncOp> {
  static Value AllocOrMapOutTensor(OpResult res, mlir::PatternRewriter& rewriter) {
    if (auto type = res.getType().dyn_cast_or_null<TensorType>()) {
      int ret_index = -1;
      for (auto use : res.getUsers()) {
        if (auto to_ret = llvm::dyn_cast_or_null<TensorToRetOp>(use)) {
          ret_index = to_ret.index();
          break;
        }
      }
      auto mem_type = MemRefType::get(type.getShape(), type.getElementType());
      auto out =
          (ret_index == -1)
              ? rewriter.create<AllocMemrefOp>(rewriter.getUnknownLoc(), mem_type)
              : rewriter.create<RetToMemrefOp>(rewriter.getUnknownLoc(), mem_type, ret_index);
      return out->getResult(0);
    }
    return nullptr;
  }
  static func::FuncOp WrapOps(func::FuncOp func, mlir::PatternRewriter& rewriter,
                              const std::string& func_name) {
    OpBuilder::InsertionGuard insertGuard(rewriter);
    auto func_type = rewriter.getFunctionType({}, {});
    rewriter.setInsertionPoint(func);
    auto wrap_func = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), func_name, func_type);
    auto& block = wrap_func.getBody().emplaceBlock();
    rewriter.setInsertionPointToStart(&block);

    auto& ops = func.getBody().front();
    BlockAndValueMapping mapper;
    for (auto& op : ops) {
      llvm::TypeSwitch<Operation*>(&op)
          .Case<ArgToTensorOp>([&](ArgToTensorOp op) {
            auto mem_type = MemRefType::get(op.getType().getShape(), op.getType().getElementType());
            auto mem_op = rewriter.create<ArgToMemrefOp>(op->getLoc(), mem_type, op.index());
            mapper.map(op, mem_op);
          })
          .Case<TensorToRetOp>([&](TensorToRetOp op) {
            auto mem_type = MemRefType::get(op.getType().getShape(), op.getType().getElementType());
            rewriter.create<MemrefToRetOp>(op->getLoc(), mem_type, mapper.lookup(op.tensor()),
                                           op.index());
          })
          .Default([&](Operation* op) {
            if (oneflow::OneFlowDialect::getDialectNamespace().equals(
                    op->getDialect()->getNamespace())) {
              // record outs type
              llvm::SmallVector<Type> mem_outs_types;
              for (auto it : op->getResultTypes()) {
                if (auto type = it.dyn_cast_or_null<TensorType>()) {
                  auto mem_type = MemRefType::get(type.getShape(), type.getElementType());
                  mem_outs_types.push_back(mem_type);
                } else {
                  llvm::errs() << "Fail to identify op type in wrap okm kernel";
                  exit(1);
                }
              }
              llvm::SmallVector<Value> map_ins;
              for (auto in : op->getOperands()) { map_ins.push_back(mapper.lookup(in)); }
              // append alloc outs after ins
              for (auto out : op->getResults()) {
                if (auto new_out = AllocOrMapOutTensor(out, rewriter)) {
                  map_ins.push_back(new_out);
                } else {
                  llvm::errs() << "Fail to alloc or map op in wrap okm kernel";
                  exit(1);
                }
              }
              // TODO: append tmp buffer after outs
              auto wrapper_op =
                  rewriter.create<WrapperOp>(op->getLoc(), mem_outs_types, ValueRange(map_ins));
              for (auto elem : llvm::zip(op->getResults(), wrapper_op->getResults())) {
                mapper.map(std::get<0>(elem), std::get<1>(elem));
              }
              auto& wrap_block = wrapper_op.body().emplaceBlock();
              OpBuilder::InsertionGuard insertGuard(rewriter);
              rewriter.setInsertionPointToStart(&wrap_block);
              ImplicitLocOpBuilder nb(rewriter.getUnknownLoc(), rewriter);
              BlockAndValueMapping wrap_mapper;
              for (auto in : llvm::zip(op->getOperands(), wrapper_op.getOperands())) {
                auto to_tensor = rewriter.create<mlir::bufferization::ToTensorOp>(
                    rewriter.getUnknownLoc(), std::get<1>(in));
                wrap_mapper.map(std::get<0>(in), to_tensor);
              }
              auto new_op = nb.clone(*op, wrap_mapper);
              SmallVector<Value> outs;
              for (auto out : new_op->getResults()) {
                if (auto type = out.getType().dyn_cast_or_null<TensorType>()) {
                  auto mem_type = MemRefType::get(type.getShape(), type.getElementType());
                  auto to_memref = rewriter.create<mlir::bufferization::ToMemrefOp>(
                      rewriter.getUnknownLoc(), mem_type, out);
                  outs.push_back(to_memref);
                } else {
                  llvm::errs() << "Fail to identify op type in wrap okm kernel";
                  exit(1);
                }
              }
              rewriter.create<ReturnOp>(rewriter.getUnknownLoc(), ValueRange(outs));
            }
          });
    }
    rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc());
    return wrap_func;
  }

  explicit WrapOKMKernelPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    const auto sym_name = op.getSymName();
    if (sym_name.startswith(func_name::MEM_GRAPH_NAME)) {
      if (op->getAttr(func_name::TAG_NAME)) { return success(); }
      // rename function
      const auto index = sym_name.substr(strlen(func_name::MEM_GRAPH_NAME)).str();
      const std::string rename = func_name::WRAP_GRAPH_NAME + index;
      // wrap kernels
      WrapOps(op, rewriter, rename);
      op->setAttr(func_name::TAG_NAME, rewriter.getBoolAttr(true));
      return success();
    }
    return failure();
  }
};

class WrapOKMKernelPass : public WrapOKMKernelPassBase<WrapOKMKernelPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<oneflow::OneFlowDialect>();
    registry.insert<OKMDialect>();
    registry.insert<bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    patterns.add<WrapOKMKernelPattern>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

std::unique_ptr<Pass> createWrapOKMKernelPass() { return std::make_unique<WrapOKMKernelPass>(); }

struct OptOKMMemrefPattern : public mlir::OpRewritePattern<func::FuncOp> {
  explicit OptOKMMemrefPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    const auto sym_name = op.getSymName();
    if (sym_name.startswith(func_name::WRAP_GRAPH_NAME)) {
      if (op->getAttr(func_name::TAG_NAME)) { return success(); }
      // rename function
      const auto index = sym_name.substr(strlen(func_name::WRAP_GRAPH_NAME)).str();
      const std::string rename = func_name::OPT_GRAPH_NAME + index;
      // opt alloc
      op->setAttr(func_name::TAG_NAME, rewriter.getBoolAttr(true));
      return success();
    }
    return failure();
  }
};

class OptOKMMemrefPass : public OptOKMMemrefPassBase<OptOKMMemrefPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<oneflow::OneFlowDialect>();
    registry.insert<OKMDialect>();
    registry.insert<bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    patterns.add<OptOKMMemrefPattern>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

std::unique_ptr<Pass> createOptOKMMemrefPass() { return std::make_unique<WrapOKMKernelPass>(); }

}  // namespace okm

}  // namespace mlir