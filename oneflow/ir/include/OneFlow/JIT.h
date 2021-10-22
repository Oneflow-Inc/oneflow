#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_JIT_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_JIT_H_

#include "mlir/IR/Value.h"
#include "oneflow/core/framework/tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "oneflow/ir/oneflow-translate/include/OneFlow/MLIROneFlowTranslation.h"

namespace oneflow {

namespace one {

namespace ir {

using namespace mlir;
using ValueMapping = std::unordered_map<Tensor*, mlir::Value>;

class JitImporter : public Importer {
 public:
  using Importer::Importer;
  ~JitImporter() = default;
  LogicalResult AppendDataInOperand(const std::string& lbn,
                                    std::vector<::mlir::Value>& operand_vec) override {
    return success();
  };
  LogicalResult AppendCtrlInOperand(const ::oneflow::OperatorConf& op,
                                    std::vector<::mlir::Value>& operand_vec) override {
    return success();
  };
  LogicalResult ProcessSystemOp(const ::oneflow::OperatorConf& op) override { return success(); };
  LogicalResult AddDeviceName(const ::oneflow::OperatorConf& op,
                              std::vector<NamedAttribute>& attr_vec) override;
  LogicalResult InsertOpResults(Operation*) override;
  Type GetTensorTypeOfLbn(const std::string& lbn) override;
  ::oneflow::AttrType QueryAttrType(const std::string& op_type_name,
                                    const std::string& attr_name) override;

 private:
  ValueMapping mapping_;
};

void MapTensorToMlirValue(Tensor* tensor, mlir::Value value, ValueMapping* mapping);
OwningOpRef<ModuleOp> CreateJitModule(MLIRContext* context);

}  // namespace ir

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_JIT_H_
