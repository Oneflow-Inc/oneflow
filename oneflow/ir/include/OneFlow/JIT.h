#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_JIT_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_JIT_H_

#include "mlir/IR/Value.h"
#include "oneflow/core/framework/arg_tuple.h"
#include "oneflow/core/framework/tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/ir/oneflow-translate/include/OneFlow/MLIROneFlowTranslation.h"
#include "oneflow/core/framework/util.h"

namespace oneflow {

namespace one {

namespace ir {

using namespace mlir;
using ValueMapping = std::unordered_map<Tensor*, mlir::Value>;
using OperandMapping = std::unordered_map<std::string, mlir::Value>;
using ResultTypeMapping = std::unordered_map<std::string, mlir::Type>;

class JitImporter : public Importer {
 public:
  using Importer::Importer;
  ~JitImporter() = default;
  LogicalResult AppendDataInOperand(const std::string& key, const int32_t index,
                                    const std::string& lbn,
                                    std::vector<::mlir::Value>& operand_vec) override;
  LogicalResult AppendCtrlInOperand(const ::oneflow::OperatorConf& op,
                                    std::vector<::mlir::Value>& operand_vec) override {
    return success();
  };
  LogicalResult ProcessSystemOp(const ::oneflow::OperatorConf& op) override { return success(); };
  LogicalResult AddDeviceName(const ::oneflow::OperatorConf& op,
                              std::vector<NamedAttribute>& attr_vec) override;
  // save tensor=>value mapping
  LogicalResult InsertOpResults(Operation*) override;
  Type GetTensorTypeOfLbn(const std::string& lbn) override;
  ::oneflow::AttrType QueryAttrType(const std::string& op_type_name,
                                    const std::string& attr_name) override;
  // 1. if func absent, create it
  // 2. if input tensor absent in tensor=>value mapping, udpate function arg
  // 3. add input to tensor=>value mapping
  // 4. insert to PlaceholderBn => value mapping (only for one op)
  mlir::FuncOp GetOrInsertFunc(const std::string& func_name, const TensorTuple& inputs,
                               TensorTuple* outputs);
  void CreateOperandMapping(const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                            const TensorTuple& inputs);
  mlir::Value GetOperandByPlaceholderBn(const std::string& bn);
  // get blob decs from inferred op
  mlir::Type GetResultTypeByBn(const std::string& bn);
  mlir::Value GetResultByBnAndIndex(const std::string& bn, const int32_t index);

 private:
  ValueMapping result_mapping_;  // tensor* => %result
  // members below should be reset every op by calling CreateMapping
  std::shared_ptr<const ArgTuple> input_arg_tuple_;
  OperandMapping operand_mapping_;  // "a0" => %result
  ResultTypeMapping output_bn_mapping_;
  TensorTuple inputs_;
  //   TensorTuple* outputs_;
};

void MapTensorToMlirValue(Tensor* tensor, mlir::Value value, ValueMapping* mapping);
OwningOpRef<ModuleOp> CreateJitModule(MLIRContext* context);

}  // namespace ir

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_JIT_H_
