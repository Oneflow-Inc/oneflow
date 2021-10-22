#include "OneFlow/JIT.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "OneFlow/OneFlowDialect.h"

namespace oneflow {

namespace one {

namespace ir {

using namespace mlir;
using ValueMapping = std::unordered_map<Tensor*, mlir::Value>;
void MapTensorToMlirValue(Tensor* tensor, mlir::Value value, ValueMapping* mapping) {
  mapping->emplace(tensor, value);
}

OwningOpRef<ModuleOp> CreateJitModule(MLIRContext* context) {
  context->loadDialect<mlir::oneflow::OneFlowDialect>();
  context->loadDialect<StandardOpsDialect>();
  OwningOpRef<ModuleOp> module(
      ModuleOp::create(FileLineColLoc::get(context, "", /*line=*/0, /*column=*/0)));
  return module;
}

LogicalResult JitImporter::AddDeviceName(const ::oneflow::OperatorConf& op,
                                         std::vector<NamedAttribute>& attr_vec) {
  return success();
}
LogicalResult JitImporter::InsertOpResults(Operation*) { return success(); }
Type JitImporter::GetTensorTypeOfLbn(const std::string& lbn) { return GetBuilder().getF128Type(); }
::oneflow::AttrType JitImporter::QueryAttrType(const std::string& op_type_name,
                                               const std::string& attr_name) {
  return ::oneflow::AttrType::kAtDataType;
}

}  // namespace ir

}  // namespace one

}  // namespace oneflow
