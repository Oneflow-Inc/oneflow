#include "generator.h"
#include <llvm/ADT/SmallVector.h>
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

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "OneFlow/OneFlowOps.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace functional = ::oneflow::one::functional;
namespace of = ::oneflow;
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
      builder.create<FrozenVariableOp>(mop->getLoc(), dense.getType(), ValueRange(), attrs);

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
  auto add01 = builder.create<BroadcastAddOp>(mop->getLoc(), result_type,
                                              ValueRange{rands[0], rands[1]}, addop_attrs);
  auto res1 = builder.create<BroadcastAddOp>(mop->getLoc(), result_type,
                                             ValueRange{add01, rands[2]}, addop_attrs);

  auto add12 = builder.create<BroadcastAddOp>(mop->getLoc(), result_type,
                                              ValueRange{rands[1], rands[2]}, addop_attrs);
  auto res2 = builder.create<BroadcastAddOp>(mop->getLoc(), result_type,
                                             ValueRange{rands[0], add12}, addop_attrs);
  // how to evaluate res1 & res2?
  // res1 and res2 equal, add to pdl
  std::cout << "generator\n";
  return;
}
}  // namespace oneflow
}  // namespace mlir

int main(/* int argc, char** argv */) {
  mlir::DialectRegistry registry;
  mlir::registerAllTranslations();
  registry.insert<mlir::oneflow::OneFlowDialect>();
  registry.getDialectAllocator("oneflow");
  // TODO: add pdl dialect
  mlir::MLIRContext context{registry};
  mlir::oneflow::Generator gen(&context);
  gen.run();
  return 0;
}
