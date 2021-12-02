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
#ifndef ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_JIT_H_
#define ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_JIT_H_

#include <utility>

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

class TensorRef final : public TensorIf<TensorRef> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorRef);
  explicit TensorRef(const std::shared_ptr<Tensor>& tensor) : tensor_(tensor) {}
  ~TensorRef() override = default;

  // Getters
  const std::shared_ptr<const Shape>& shape() const override { return tensor_->shape(); }
  Symbol<DType> dtype() const override { return tensor_->dtype(); }
  Maybe<TransportToken> transport_token() const override { return tensor_->transport_token(); }
  Maybe<Symbol<cfg::NdSbp>> nd_sbp() const override { return tensor_->nd_sbp(); }
  Maybe<Symbol<ParallelDesc>> parallel_desc() const override { return tensor_->parallel_desc(); }
  Maybe<Symbol<Device>> device() const override { return tensor_->device(); }
  Maybe<Symbol<Device>*> mut_device() override { return tensor_->mut_device(); }
  bool is_lazy() const override { return tensor_->is_lazy(); }
  bool is_consistent() const override { return tensor_->is_consistent(); }
  bool is_cuda() const override { return tensor_->is_cuda(); };
  const TensorMeta& tensor_meta() const override { return tensor_->tensor_meta(); }
  Maybe<Tensor> data() override { return tensor_->data(); }

  // Getters valid only for EagerMirroredTensor
  Maybe<vm::EagerBlobObject> eager_blob_object() const override {
    return tensor_->eager_blob_object();
  }
  Maybe<LocalDepObject*> compute_local_dep_object() const override {
    return tensor_->compute_local_dep_object();
  }
  Maybe<TensorStorage> tensor_storage() const override { return tensor_->tensor_storage(); }
  Maybe<bool> has_eager_blob_object() const override { return tensor_->has_eager_blob_object(); }
  Maybe<const Stride> stride() const override { return tensor_->stride(); }
  Maybe<int64_t> storage_offset() const override { return tensor_->storage_offset(); }

  // Getters for autograd
  Maybe<Tensor> acc_grad() const override { return tensor_->acc_grad(); }
  Maybe<TensorArg> current_grad() const override { return tensor_->current_grad(); }
  bool requires_grad() const override { return tensor_->requires_grad(); }
  bool is_leaf() const override { return tensor_->is_leaf(); }
  bool retain_grad() const override { return tensor_->retain_grad(); }
  bool has_autograd_meta() const override { return tensor_->has_autograd_meta(); }

  // Setters for autograd
  Maybe<void> set_acc_grad(const std::shared_ptr<Tensor>& grad) override {
    return tensor_->set_acc_grad(grad);
  }
  Maybe<void> set_requires_grad(bool requires_grad) override {
    return tensor_->set_requires_grad(requires_grad);
  }
  Maybe<void> set_retain_grad(bool retain_grad) override {
    return tensor_->set_retain_grad(retain_grad);
  }
  Maybe<Tensor> mut_acc_grad() override { return tensor_->mut_acc_grad(); }
  void set_is_leaf(bool is_leaf) override { tensor_->set_is_leaf(is_leaf); }
  std::shared_ptr<AutogradMeta> mut_autograd_meta() override {
    return tensor_->mut_autograd_meta();
  }
  void set_autograd_meta(const std::shared_ptr<AutogradMeta>& autograd_meta) override {
    tensor_->set_autograd_meta(autograd_meta);
  }

  // Operators for tensor
  Maybe<Tensor> detach() const override { return tensor_->detach(); }
  Maybe<Tensor> clone() const override { return tensor_->clone(); }
  Maybe<EagerMirroredTensorImpl*> mut_eager_mirrored_tensor_impl() override {
    return tensor_->mut_eager_mirrored_tensor_impl();
  }
  user_op::TensorDesc* mut_tensor_meta() override { return tensor_->mut_tensor_meta(); }
  Maybe<void> set_data(const std::shared_ptr<Tensor>& other) override {
    return tensor_->set_data(other);
  }

  Maybe<MirroredTensor> AsMirroredTensor() override { return tensor_->AsMirroredTensor(); }
  Maybe<ConsistentTensor> AsConsistentTensor() override { return tensor_->AsConsistentTensor(); }

  void ResetTensor(const std::shared_ptr<Tensor>& tensor) { tensor_ = tensor; }
  std::shared_ptr<Tensor> GetTensor() { return tensor_; }

 private:
  std::shared_ptr<Tensor> tensor_;
};

class ProcessOpContext {
 public:
  ProcessOpContext() = default;
  ProcessOpContext(const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                   const TensorTuple& inputs, TensorTuple* outputs,
                   std::shared_ptr<const ParallelDesc> parallel_desc)
      : input_arg_tuple_(input_arg_tuple),
        inputs_(inputs),
        outputs_(outputs),
        parallel_desc_(std::move(parallel_desc)) {}
  std::shared_ptr<const ParallelDesc> GetParallelDesc() const {
    CHECK(parallel_desc_);
    return parallel_desc_;
  }
  void SetParallelDesc(const std::shared_ptr<const ParallelDesc>& parallel_desc) {
    parallel_desc_ = parallel_desc;
  }
  std::shared_ptr<const ArgTuple> GetInputArgTuple() const { return input_arg_tuple_; }
  const TensorTuple& GetInputs() const { return inputs_; }
  TensorTuple* GetOutputs() { return outputs_; }
  std::unordered_map<std::string, mlir::Value>& GetOperandMapping() { return operand_mapping_; }
  std::unordered_map<std::string, mlir::Type>& GetResultTypeMapping() {
    return result_type_mapping_;
  }

 private:
  std::shared_ptr<const ArgTuple> input_arg_tuple_;
  TensorTuple inputs_;
  TensorTuple* outputs_{};
  std::unordered_map<std::string, mlir::Value> operand_mapping_;     // "a0" => %result
  std::unordered_map<std::string, mlir::Type> result_type_mapping_;  // "a0" => tensor<2x2xf32>
  std::shared_ptr<const ParallelDesc> parallel_desc_;
};

class ProcessFuncContext {
 public:
  ProcessFuncContext() = default;
  ProcessFuncContext(ProcessFuncContext&&) = default;
  ProcessFuncContext(const ProcessFuncContext&) = default;
  ProcessFuncContext& operator=(ProcessFuncContext&&) = default;
  ProcessFuncContext& operator=(const ProcessFuncContext&) = default;
  ~ProcessFuncContext() = default;

  void InsertValueMapping(Tensor*, mlir::Value);
  mlir::Value GetValue(Tensor*);
  const llvm::DenseMap<Tensor*, mlir::Value>& GetValueMapping();
  void InsertIntermediateTensor(Value, std::shared_ptr<TensorRef>);
  const llvm::DenseMap<Value, std::shared_ptr<TensorRef>>& GetIntermediateTensorsMapping() {
    return intermediate_tensors_mapping_;
  }

 private:
  llvm::DenseMap<Tensor*, mlir::Value> value_mapping_;
  llvm::DenseMap<Value, std::shared_ptr<TensorRef>> intermediate_tensors_mapping_;
};

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
  LogicalResult AppendCtrlOutType(llvm::SmallVector<Type, 8>& out_types) override {
    return failure();
  }
  LogicalResult ProcessSystemOp(const ::oneflow::OperatorConf& op) override { return success(); };
  LogicalResult AddDeviceName(const ::oneflow::OperatorConf& op,
                              std::vector<NamedAttribute>& attr_vec) override;
  // save tensor=>value mapping
  LogicalResult InsertOpResults(const ::oneflow::OperatorConf& op, Operation*) override;
  Type GetTensorTypeOfLbn(const std::string& lbn) override;
  // 1. if func absent, create it
  // 2. if input tensor absent in tensor=>value mapping, udpate function arg
  // 3. add input to tensor=>value mapping
  // 4. insert to PlaceholderBn => value mapping (only for one op)
  mlir::FuncOp GetOrInsertFunc(const std::string& func_name);
  void CreateOperandMapping(const ::oneflow::OperatorConf& op,
                            const std::shared_ptr<const ParallelDesc>,
                            const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                            const TensorTuple& inputs, TensorTuple* outputs);
  // get blob decs from inferred op

  llvm::Optional<mlir::Value> GetResultByBnAndIndex(const std::string& bn, const int32_t index);
  std::shared_ptr<Tensor> MakeIntermediateTensor(
      const std::string& lbn, Value result,
      const std::shared_ptr<const ParallelDesc>& parallel_desc);
  llvm::Optional<TensorType> GetMlirTensorTypeFromBlobDesc(const BlobDesc& blob_desc);
  void SetParallelDesc(const std::shared_ptr<const ParallelDesc>& parallel_desc) {
    GetProcessOpContext().SetParallelDesc(parallel_desc);
  }
  LogicalResult FinalizeProcessFunction();
  ProcessOpContext& GetProcessOpContext() { return process_op_context_; }
  const llvm::DenseMap<Value, std::shared_ptr<TensorRef>>& GetIntermediateTensorsMapping() {
    return process_func_context_.GetIntermediateTensorsMapping();
  }
  const llvm::DenseMap<Tensor*, mlir::Value>& GetValueMapping() {
    return process_func_context_.GetValueMapping();
  }
  void SaveIntermediate(Value v, std::shared_ptr<TensorRef> r) {
    process_func_context_.InsertIntermediateTensor(v, std::move(r));
  }
  void TrackTensorAndValue(Tensor* t, mlir::Value v) {
    process_func_context_.InsertValueMapping(t, v);
  }

 private:
  // reset every func
  ProcessFuncContext process_func_context_;
  // reset every op
  ProcessOpContext process_op_context_;
  // persistent
  DenseMap<llvm::hash_code, std::string> func_hash_symbol_mapping_;
};

OwningOpRef<ModuleOp> CreateJitModule(MLIRContext* context);

}  // namespace ir

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_JIT_H_
