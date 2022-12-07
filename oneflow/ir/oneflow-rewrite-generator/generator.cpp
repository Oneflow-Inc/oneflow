#include "generator.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "wrapper.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>
#include <algorithm>
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowSupport.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "OneFlow/OneFlowOps.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/ir/oneflow-translate/include/OneFlow/MLIROneFlowTranslation.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/PDL/IR/PDLOps.h.inc"
#include "mlir/Dialect/PDL/IR/PDL.h"

namespace functional = ::oneflow::one::functional;
namespace of = ::oneflow;
namespace pdl = ::mlir::pdl;

namespace llvm {}
namespace mlir {
namespace oneflow {

auto Generator::get_random_tensor() {
  // get random i64 tensor, run on cpu?
  auto rand =
      functional::RandN({2, 3}, of::DType{of::kInt64}, of::Device::ParseAndNew("cpu").GetOrThrow(),
                        of::one::DefaultCPUGenerator().GetOrThrow(), false)
          .GetPtrOrThrow();
  int64_t rand_arr[256];
  const auto& callback = [&](of::ep::Stream* stream,
                             const std::shared_ptr<of::vm::EagerBlobObject>& eager_blob_object) {
    of::AutoMemcpy(stream, &rand_arr, eager_blob_object->dptr(), 2 * 3 * sizeof(int64_t),
                   of::memory::MakeHostMemCase(), eager_blob_object->mem_case());
  };
  std::cout << rand_arr[0] << ", " << rand_arr[1];
  CHECK_JUST(of::one::SyncAccessTensorWithTimeOut(rand, callback, "const"));
  auto dense = support::TensorToDenseElementsAttr(rand, context);
  dense.print(llvm::outs());
  NamedAttrList attrs;
  attrs.set("value", dense);

  auto frozen =
      builder.create<FrozenVariableOp>(graph->getLoc(), dense.getType(), ValueRange(), attrs);

  return frozen;
}

void Generator::run() {
  OpBuilder builder(context);
  llvm::SmallVector<FrozenVariableOp, 4> rands(4);
  std::generate(rands.begin(), rands.end(), [this]() { return get_random_tensor(); });
  // MLIR: attrs and operands difference?
  // addop_attrs.set("x", r);
  // addop_attrs.set("y", r);
  NamedAttrList addop_attrs;
  auto result_type = rands[0].getType();
  auto loc = graph->getLoc();
  auto add01 =
      builder.create<BroadcastAddOp>(loc, result_type, ValueRange{rands[0], rands[1]}, addop_attrs);
  auto res1 = builder.create<BroadcastAddOp>(add01->getLoc(), result_type,
                                             ValueRange{add01, rands[2]}, addop_attrs);

  auto add12 = builder.create<BroadcastAddOp>(res1->getLoc(), result_type,
                                              ValueRange{rands[1], rands[2]}, addop_attrs);
  auto res2 = builder.create<BroadcastAddOp>(add12->getLoc(), result_type,
                                             ValueRange{rands[0], add12}, addop_attrs);
  // TODO: how to evaluate res1 & res2?
  // I have a job, but the job is constituted of multiple OPs
  of::Job job;
  of::RoundTripOneFlowJobWrapper<of::kBeforeAD> job_wrapper(&job);
  JobImporter importer(job_wrapper, context, graph);
  if (importer.TryToUpdateJob().succeeded())
    std::cout << "yeah!\n";
  else
    std::cout << "no~~\n";
  auto opconf = job.net().op(1);
  for (auto& op : job.net().op()) {
    auto& user_op_conf = op.user_conf();
    user_op_conf.input();  // not sure we need this
  }
  // LoadJobFromIR()
  // check their hash (and that the hashes are equal)

  // res1 and res2 equal, add to pdl
  TypeAttr ta;
  auto pdl_result_type = builder.create<pdl::TypeOp>(pdl.getModule().getLoc(), result_type, ta);
  auto pdl_op1 = builder.create<pdl::OperandOp>(pdl_result_type.getLoc());
  auto pdl_op2 = builder.create<pdl::OperandOp>(pdl_op1.getLoc());
  auto pdl_op3 = builder.create<pdl::OperandOp>(pdl_op2.getLoc());
  auto pdl_of_add_op = builder.create<pdl::OperationOp>(pdl_op3->getLoc(), res1.op_name(),
                                                        ValueRange{pdl_op1, pdl_op2});
  auto pdl_res1 = builder.create<pdl::OperationOp>(pdl_of_add_op->getLoc(), res1.op_name(),
                                                   ValueRange{pdl_of_add_op, pdl_op3});

  auto pdl_of_add_op2 = builder.create<pdl::OperationOp>(pdl_res1->getLoc(), res1.op_name(),
                                                         ValueRange{pdl_op2, pdl_op3});
  auto pdl_res2 = builder.create<pdl::OperationOp>(pdl_of_add_op2->getLoc(), res1.op_name(),
                                                   ValueRange{pdl_op1, pdl_of_add_op2});
  auto rewrite =
      builder.create<pdl::RewriteOp>(pdl_res2->getLoc(), pdl_res1, nullptr, ValueRange());
  // final replace op
  builder.create<pdl::ReplaceOp>(rewrite->getLoc(), pdl_res1, pdl_res2, ValueRange());
}

size_t Generator::fingerprint(Operation* op) {
  // TODO: get forward result, get hash
  return {};
}

void Generator::dfs(int depth, SmallVector<Value*>& inputs) {
  auto res = graph.walk([](Operation* op) {
    // add op to some set, check existence
    op->print(llvm::outs());
    return WalkResult::advance();
  });
  if (res.wasInterrupted())  // contains duplicate operations
    return;
  auto graph_pdl = build_pdl_from_oneflow_op(graph);
  D.insert({fingerprint(graph), graph_pdl});
  if (depth > 3) return;
  dfs_broadcast_binary_ops<
#define GET_OP_LIST
#include "OneFlow/OneFlow.broadcast_ops.cpp.inc"
      >(depth, inputs);
  // and binary ops, conv ops, math ops, matmul ops, etc.
  // how to check validity of Operation?
  // use verifier, is that enough?
}

ModuleOp Generator::build_pdl_from_oneflow_op(Operation* op) {
  static int graph_index{};
  // go over the operands
  BlockAndValueMapping bav{};
  auto new_pdl_module = ModuleOp::create(FileLineColLoc::get(
      builder.getStringAttr("pdl-" + std::to_string(graph_index++) + ".mlir"), 0, 0));
  auto loc = new_pdl_module.getLoc();
  // clone graph to the new pdl module
  assert(graph->getNumRegions() == 1);
  for (Block& block : graph.getBodyRegion().getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (llvm::dyn_cast<FrozenVariableOp>(op)) {
        // create pdl operand op
        auto pdlop = builder.create<pdl::OperandOp>(loc);
        bav.map(op.getResult(0), pdlop);  // can this compile?
        loc = pdlop->getLoc();
      } else {
        // normal op, create operation op
        SmallVector<Value> pdl_operands;
        for (auto operand : op.getOperands()) { pdl_operands.push_back(bav.lookup(operand)); }
        auto pdlop =
            builder.create<pdl::OperationOp>(loc, op.getName(), pdl_operands, op.getAttrs());
        // what if the original op has multiple results? pdl OperationOp is OneResult??
        for (auto result : op.getResults()) { bav.map(result, pdlop); }
        loc = pdlop->getLoc();
      }
    }
  }
  return new_pdl_module;
}

template<typename... Args>
void Generator::dfs_broadcast_binary_ops(int depth, SmallVector<Value*>& inputs) {
  /**
   * for i ∈ inputs and i is a valid input to op do
   * Add operator op into graph G.
   * Add the output tensors of op into I.
   */
  (void)std::initializer_list<int>{0, (dfs_broadcast_binary_op<Args>(depth + 1, inputs), 0)...};
  // taken from Dialect::addOperations
}

template<typename T>
void Generator::dfs_broadcast_binary_op(int depth, SmallVector<Value*>& inputs) {
  inputs.reserve(inputs.size() + 1);
  for (auto it1 = inputs.begin(); it1 != inputs.end(); ++it1) {
    for (auto it2 = it1; it2 != inputs.end(); ++it2) {
      auto op = builder.create<T>(graph->getLoc(), TypeRange((*it1)->getType()),
                                  ValueRange({**it1, **it2}));
      if (op.verifyInvariants().succeeded()) {
        for (OpResult out : op->getOpResults()) { inputs.push_back(&out); }
        // go down one level
        dfs(depth + 1, inputs);
        // remove op results from input
        inputs.pop_back_n(op->getOpResults().size());
        // delete op from graph, how to delete??
      } else {
        return;
      }
    }
  }
}
}  // namespace oneflow
}  // namespace mlir

int main(/* int argc, char** argv */) {
  mlir::DialectRegistry registry;
  mlir::registerAllTranslations();
  registry.insert<mlir::oneflow::OneFlowDialect>();
  registry.insert<mlir::pdl::PDLDialect>();
  registry.getDialectAllocator("oneflow");
  // TODO: add pdl dialect
  mlir::MLIRContext context{registry};
  mlir::oneflow::Generator gen(&context);
  gen.run();
  return 0;
}
