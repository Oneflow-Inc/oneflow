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

LogicalResult JitImporter::AppendDataInOperand(const std::string& lbn,
                                               std::vector<::mlir::Value>& operand_vec) {
  llvm::errs() << "[AppendDataInOperand] " << lbn << "\n";
  return success();
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

mlir::FuncOp JitImporter::GetOrInsertFuncAndCreateMapping(const std::string& func_name,
                                                          const TensorTuple& inputs,
                                                          TensorTuple* outputs) {
  // convert data types from oneflow
  auto arg_types = llvm::SmallVector<Type, 8>();
  auto result_types = llvm::SmallVector<Type, 8>();
  for (const auto& input : inputs) {
    auto mlir_type = GetTypeFromOneFlowDataType(input->dtype()->data_type());
    arg_types.push_back(mlir_type.getValue());
  }
  for (const auto& output : *outputs) {
    auto mlir_type = GetTypeFromOneFlowDataType(output->dtype()->data_type());
    result_types.push_back(mlir_type.getValue());
  }
  // found existing func or create new one
  SymbolTable symbol_table(GetModule());
  FuncOp found_func = symbol_table.lookup<FuncOp>(func_name);
  if (found_func) {
    return found_func;
  } else {
    auto func_type = GetBuilder().getFunctionType(arg_types, llvm::NoneType());
    FuncOp function = mlir::FuncOp::create(GetRootLocation(), func_name, func_type);
    auto entryBlock = function.addEntryBlock();
    GetBuilder().setInsertionPointToStart(entryBlock);
    GetModule().push_back(function);
    return function;
  }
}
}  // namespace ir

}  // namespace one

}  // namespace oneflow
