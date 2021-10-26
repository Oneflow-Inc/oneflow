#include "OneFlow/JIT.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "OneFlow/OneFlowDialect.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace one {

namespace ir {

using namespace mlir;

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
  operand_vec.push_back(GetResultByBnAndIndex(key, index));
  return success();
}
LogicalResult JitImporter::AddDeviceName(const ::oneflow::OperatorConf& op,
                                         std::vector<NamedAttribute>& attr_vec) {
  return success();
}
Type JitImporter::GetTensorTypeOfLbn(const std::string& lbn) {
  llvm::errs() << "[GetTensorTypeOfLbn] " << lbn << "\n";
  LogicalBlobId lbi = GenLogicalBlobId(lbn);
  return result_type_mapping_.at(lbi.blob_name());
}
std::shared_ptr<MirroredTensor> JitImporter::MakeIntermediateTensor(
    const std::string& lbn, Value result,
    const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  auto tensor_type = result.getType().cast<TensorType>();
  auto dtype = DataType::kInvalidDataType;
  if (tensor_type.getElementType().isF32()) {
    dtype = DataType::kFloat;
  } else {
    result.dump();
    LOG(FATAL) << "fail to creat tensor";
  }
  const auto& device = CHECK_JUST(Device::MakeDeviceByParallelDesc(*parallel_desc));
  auto shape_from_mlir = new Shape({tensor_type.getShape().begin(), tensor_type.getShape().end()});
  auto shape = std::make_shared<Shape>();
  shape.reset(shape_from_mlir);
  auto tensor = MirroredTensor::MakeTensor(shape, dtype, device, /* is_lazy */ true,
                                           /* requires_grad= */ false, /* is_leaf= */ true)
                    .GetPtrOrThrow();
  CHECK(intermediate_tensors_.emplace(lbn, tensor).second)
      << "Intermediate tensor already created, lbn: " << lbn;
  CHECK(result_mapping_.emplace(tensor.get(), result).second)
      << "Intermediate tensor already mapped to mlir value, lbn: " << lbn;
  return tensor;
}
LogicalResult JitImporter::InsertOpResults(const ::oneflow::OperatorConf& op_conf,
                                           Operation* created_op) {
  auto output_lbns = created_op->getAttrOfType<ArrayAttr>("output_lbns");
  CHECK_EQ(output_lbns.size(), outputs_->size());
  for (auto data_out : llvm::enumerate(GetDataOutputResults(created_op))) {
    auto lbn = output_lbns[data_out.index()].dyn_cast<StringAttr>().getValue().str();
    auto tensor = MakeIntermediateTensor(lbn, data_out.value(), parallel_desc_);
    (*outputs_)[data_out.index()] = tensor;
  }
  return success();
}
::oneflow::AttrType JitImporter::QueryAttrType(const std::string& op_type_name,
                                               const std::string& attr_name) {
  return ::oneflow::AttrType::kAtDataType;
}

mlir::FuncOp JitImporter::GetOrInsertFunc(const std::string& func_name, const TensorTuple& inputs,
                                          TensorTuple* outputs) {
  // convert data types from oneflow
  outputs_ = outputs;
  auto result_types = llvm::SmallVector<Type, 8>();
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

std::unique_ptr<BlobDesc> GetBlobDescFromMlirTensorType(TensorType tensor_type) {
  auto dtype = DataType::kInvalidDataType;
  if (tensor_type.getElementType().isF32()) {
    dtype = DataType::kFloat;
  } else {
    tensor_type.dump();
    LOG(FATAL) << "fail to get BlobDesc from TensorType";
  }
  auto shape_from_mlir = new Shape({tensor_type.getShape().begin(), tensor_type.getShape().end()});
  return std::make_unique<BlobDesc>(*shape_from_mlir, dtype);
}

llvm::Optional<TensorType> JitImporter::GetMlirTensorTypeFromBlobDesc(const BlobDesc& blob_desc) {
  if (auto t = GetTypeFromOneFlowDataType(blob_desc.data_type())) {
    return RankedTensorType::get(
        ArrayRef<int64_t>(blob_desc.shape().dim_vec().begin(), blob_desc.shape().dim_vec().end()),
        t.getValue());
  } else {
    return llvm::None;
  }
}

void JitImporter::CreateOperandMapping(const ::oneflow::OperatorConf& op_conf,
                                       const std::shared_ptr<const ParallelDesc> parallel_desc,
                                       const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                                       const TensorTuple& inputs) {
  operand_mapping_.clear();
  input_arg_tuple_ = input_arg_tuple;
  inputs_ = inputs;
  HashMap<std::string, std::unique_ptr<BlobDesc>> lbi2logical_blob_desc_;
  auto op = CHECK_JUST(ConstructOp(op_conf));
  // TODO: refactor using GetResultByBnAndIndex
  for (auto pair : llvm::zip(input_arg_tuple->indexed_bns(), inputs)) {
    const auto& indexed_bn = std::get<0>(pair);
    const auto& tensor = std::get<1>(pair);
    auto result_it = result_mapping_.find(tensor.get());
    if (result_it == result_mapping_.end()) {
      for (auto kv : result_mapping_) {
        std::string result_str;
        llvm::raw_string_ostream os(result_str);
        kv.second.print(os);
        LOG(ERROR) << "tensor/value: " << kv.first << "/" << result_str;
      }
      LOG(FATAL) << "result not found, indexed_bn: " << indexed_bn << ", tensor: " << tensor.get()
                 << ", shape: " << tensor->shape()->DebugStr()
                 << ", dtype: " << tensor->dtype()->name();
    } else {
      assert(operand_mapping_.emplace(indexed_bn, result_mapping_.at(tensor.get())).second);
    }
  }
  // TODO: refine here
  auto GetLogicalBlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
    LOG(ERROR) << "[GetLogicalBlobDesc4BnInOp] " << bn;
    if (lbi2logical_blob_desc_.find(bn) == lbi2logical_blob_desc_.end()) {
      auto operand_it = operand_mapping_.find(bn);
      if (operand_it == operand_mapping_.end()) {
        auto blob_desc = std::make_unique<BlobDesc>(DataType::kInvalidDataType);
        CHECK(lbi2logical_blob_desc_.emplace(bn, std::move(blob_desc)).second);
      } else {
        auto found = GetBlobDescFromMlirTensorType(operand_it->second.getType().cast<TensorType>());
        CHECK(lbi2logical_blob_desc_.emplace(bn, std::move(found)).second);
      }
    }
    return lbi2logical_blob_desc_.at(bn).get();
  };
  CHECK_JUST(op->InferLogicalOutBlobDescs(GetLogicalBlobDesc4BnInOp, *parallel_desc));
  for (auto& kv : lbi2logical_blob_desc_) {
    CHECK(
        result_type_mapping_.emplace(kv.first, GetMlirTensorTypeFromBlobDesc(*kv.second).getValue())
            .second);
  }
}

mlir::Value JitImporter::GetResultByBnAndIndex(const std::string& bn, const int32_t index) {
  auto idx = input_arg_tuple_->TensorTupleIndex4ArgNameAndIndex(bn, index);
  auto tensor = inputs_[idx];
  auto result_it = result_mapping_.find(tensor.get());
  if (result_it == result_mapping_.end()) {
    for (auto kv : result_mapping_) {
      std::string result_str;
      llvm::raw_string_ostream os(result_str);
      kv.second.print(os);
      LOG(ERROR) << "tensor/value: " << kv.first << "/" << result_str;
    }
    LOG(FATAL) << "result not found, arg/index: " << bn << "/" << index
               << ", tensor: " << tensor.get() << ", shape: " << tensor->shape()->DebugStr()
               << ", dtype: " << tensor->dtype()->name();
  } else {
    return result_it->second;
  }
}

}  // namespace ir

}  // namespace one

}  // namespace oneflow
