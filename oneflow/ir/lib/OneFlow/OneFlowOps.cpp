#include "OneFlow/OneFlowOps.h"
#include <iostream>
#include <string>
#include "OneFlow/OneFlowDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::oneflow;

static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes)
      || parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

static mlir::LogicalResult verify(ConstantOp op) { return mlir::success(); }

struct ConcreteUserOps : public mlir::OpRewritePattern<oneflow::UserOp> {
  ConcreteUserOps(mlir::MLIRContext *context)
      : OpRewritePattern<oneflow::UserOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(oneflow::UserOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    if (op->getAttrOfType<StringAttr>("op_type_name").getValue().equals("relu")) {
      if (op.ctrl_inputs().empty() && op.ctrl_output().use_empty()) {
        NamedAttrList attributes(op->getAttrDictionary());
        attributes.erase("operand_segment_sizes");
        attributes.erase("result_segment_sizes");
        if (auto relu = rewriter.create<oneflow::ReluOp>(
                op->getLoc(), op->getOperands().take_front(), attributes)) {
          op.data_output().front().replaceAllUsesWith(relu.y());
          op->erase();
          return success();
        }
      }
    } else /* trim redundant control outputs */ {
      if (op.ctrl_output() && op.ctrl_output().use_empty()) {
        const int32_t num_data_inputs =
            op.result_segment_sizes().getValue<IntegerAttr>({0}).getInt();
        NamedAttrList attributes(op->getAttrDictionary());
        attributes.erase("result_segment_sizes");
        attributes.append("result_segment_sizes", rewriter.getI32VectorAttr({num_data_inputs, 0}));
        if (auto sys = rewriter.create<oneflow::UserOp>(
                op->getLoc(), op.getResultTypes().take_front(op.data_output().size()),
                op->getOperands(), attributes)) {
          for (auto out : op.data_output()) {
            out.replaceAllUsesWith(sys->getResult(out.getResultNumber()));
          }
          op->erase();
          return success();
        }
      }
    }
    return failure();
  }
};

void UserOp::getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results,
                                         ::mlir::MLIRContext *context) {
  results.insert<ConcreteUserOps>(context);
}

#include "OneFlow/OneFlowEnums.cpp.inc"

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.cpp.inc"
