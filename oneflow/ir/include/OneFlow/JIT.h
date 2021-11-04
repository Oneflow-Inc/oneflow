/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_JIT_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_JIT_H_

#include "mlir/IR/Value.h"
#include "oneflow/core/framework/arg_tuple.h"
#include "oneflow/core/framework/tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/ir/oneflow-translate/include/OneFlow/MLIROneFlowTranslation.h"
#include "oneflow/core/framework/util.h"
#include "mlir/Pass/Pass.h"
#include "oneflow/core/framework/op_kernel.h"

namespace mlir {
namespace oneflow {

std::unique_ptr<Pass> createReturnAllLeaveResultPass();
std::unique_ptr<Pass> createCreateComputeCtxPass();

}  // namespace oneflow
}  // namespace mlir

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "OneFlow/OneFlowJITPasses.h.inc"

namespace oneflow {

namespace one {

namespace ir {

using namespace mlir;

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
  LogicalResult InsertOpResults(const ::oneflow::OperatorConf& op, Operation*) override;
  Type GetTensorTypeOfLbn(const std::string& lbn) override;
  ::oneflow::AttrType QueryAttrType(const std::string& op_type_name,
                                    const std::string& attr_name) override;
  // 1. if func absent, create it
  // 2. if input tensor absent in tensor=>value mapping, udpate function arg
  // 3. add input to tensor=>value mapping
  // 4. insert to PlaceholderBn => value mapping (only for one op)
  mlir::FuncOp GetOrInsertFunc(const std::string& func_name, const TensorTuple& inputs,
                               TensorTuple* outputs);
  void CreateOperandMapping(const ::oneflow::OperatorConf& op,
                            const std::shared_ptr<const ParallelDesc>,
                            const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                            const TensorTuple& inputs);
  // get blob decs from inferred op

  llvm::Optional<mlir::Value> GetResultByBnAndIndex(const std::string& bn, const int32_t index);
  std::shared_ptr<MirroredTensor> MakeIntermediateTensor(
      const std::string& lbn, Value result,
      const std::shared_ptr<const ParallelDesc>& parallel_desc);
  llvm::Optional<TensorType> GetMlirTensorTypeFromBlobDesc(const BlobDesc& blob_desc);
  void SetParallelDesc(const std::shared_ptr<const ParallelDesc>& parallel_desc) {
    parallel_desc_ = parallel_desc;
  }
  LogicalResult LowerToOneFlowKernel();
  LogicalResult WalkModuleByEveryOpConf(const std::function<void(const OperatorConf&)> cb);

 private:
  std::unordered_map<Tensor*, mlir::Value> result_mapping_;  // tensor* => %result
  // JIT interpreter owns the intermediate tensors
  // An intermediate tensor will be materialized if:
  // 1. it is a result tensor
  // 2. it is being evaluated before forward function returning (print, etc)
  std::unordered_map<std::string, std::shared_ptr<MirroredTensor>> intermediate_tensors_;
  // members below should be reset every op by calling CreateMapping
  std::shared_ptr<const ArgTuple> input_arg_tuple_;
  std::unordered_map<std::string, mlir::Value> operand_mapping_;     // "a0" => %result
  std::unordered_map<std::string, mlir::Type> result_type_mapping_;  // "a0" => tensor<2x2xf32>
  TensorTuple inputs_;
  std::shared_ptr<Operator> op_;
  TensorTuple* outputs_;
  std::shared_ptr<const ParallelDesc> parallel_desc_;
};

OwningOpRef<ModuleOp> CreateJitModule(MLIRContext* context);

}  // namespace ir

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_JIT_H_
