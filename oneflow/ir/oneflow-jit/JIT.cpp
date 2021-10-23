#include "OneFlow/JIT.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "OneFlow/OneFlowDialect.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

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

LogicalResult JitImporter::AppendDataInOperand(const std::string& key, const int32_t index,
                                               const std::string& lbn,
                                               std::vector<::mlir::Value>& operand_vec) {
  LOG(ERROR) << "[getting]" << key << "/" << index;
  operand_vec.push_back(operand_mapping_.at(std::make_pair(key, index)));
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

mlir::FuncOp JitImporter::GetOrInsertFunc(const std::string& func_name, const TensorTuple& inputs,
                                          TensorTuple* outputs) {
  // convert data types from oneflow
  auto result_types = llvm::SmallVector<Type, 8>();
  // for (const auto& output : *outputs) {
  //   auto mlir_type = GetTypeFromOneFlowDataType(output->dtype()->data_type());
  //   result_types.push_back(mlir_type.getValue());
  // }
  // found existing func or create new one
  SymbolTable symbol_table(GetModule());
  FuncOp found_func = symbol_table.lookup<FuncOp>(func_name);
  if (found_func) {
    return found_func;
  } else {
    auto arg_tensors = GetJitForwardArgs();
    auto arg_types = llvm::SmallVector<Type, 8>();
    for (const auto& arg_tensor : arg_tensors) {
      auto mlir_dtype = GetTypeFromOneFlowDataType(arg_tensor->dtype()->data_type());
      auto mlir_tensor_type =
          RankedTensorType::get(ArrayRef<int64_t>(arg_tensor->shape()->dim_vec().begin(),
                                                  arg_tensor->shape()->dim_vec().end()),
                                mlir_dtype.getValue());
      arg_types.push_back(mlir_tensor_type);
    }
    auto func_type = GetBuilder().getFunctionType(arg_types, llvm::NoneType());
    FuncOp function = mlir::FuncOp::create(GetRootLocation(), func_name, func_type);
    auto entryBlock = function.addEntryBlock();
    CHECK_EQ(arg_tensors.size(), function.body().getArguments().size());
    for (auto argument_pair : llvm::zip(arg_tensors, function.body().getArguments())) {
      CHECK(result_mapping_.emplace(std::get<0>(argument_pair).get(), std::get<1>(argument_pair))
                .second);
    }
    GetBuilder().setInsertionPointToStart(entryBlock);
    GetModule().push_back(function);
    function->dump();
    return function;
  }
}

void JitImporter::CreateOperandMapping(
    const std::vector<std::pair<std::string, int32_t>>& indexed_arg_name_and_index,
    const TensorTuple& inputs) {
  operand_mapping_.clear();
  for (auto pair : llvm::zip(indexed_arg_name_and_index, inputs)) {
    const auto& arg_name_index_tuple = std::get<0>(pair);
    const auto& tensor = std::get<1>(pair);
    auto result_it = result_mapping_.find(tensor.get());
    if (result_it == result_mapping_.end()) {
      for (auto kv : result_mapping_) {
        std::string result_str;
        llvm::raw_string_ostream os(result_str);
        kv.second.print(os);
        LOG(ERROR) << "tensor/value: " << kv.first << "/" << result_str;
      }
      LOG(FATAL) << "result not found, arg/index: " << arg_name_index_tuple.first << "/"
                 << arg_name_index_tuple.second << ", tensor: " << tensor.get()
                 << ", shape: " << tensor->shape()->DebugStr()
                 << ", dtype: " << tensor->dtype()->name();
    } else {
      assert(
          operand_mapping_.emplace(arg_name_index_tuple, result_mapping_.at(tensor.get())).second);
    }
  }
}

}  // namespace ir

}  // namespace one

}  // namespace oneflow
