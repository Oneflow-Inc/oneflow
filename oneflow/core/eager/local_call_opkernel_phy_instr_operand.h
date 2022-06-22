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
#ifndef ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_

#include "oneflow/core/vm/phy_instr_operand.h"
#include "oneflow/core/eager/dev_vm_dep_object_consume_mode.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_interpreter.h"

namespace oneflow {
namespace one {

class StatefulLocalOpKernel;
class ConsistentTensorInferResult;

using EagerBlobObjectList = std::vector<std::shared_ptr<vm::EagerBlobObject>>;
using EagerBlobObjectListPtr =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;

}  // namespace one

namespace user_op {

class OpKernel;

}  // namespace user_op

namespace vm {

class LocalCallOpKernelPhyInstrOperand : public vm::PhyInstrOperand {
 public:
  LocalCallOpKernelPhyInstrOperand(const LocalCallOpKernelPhyInstrOperand&) = delete;
  LocalCallOpKernelPhyInstrOperand(LocalCallOpKernelPhyInstrOperand&&) = delete;
  ~LocalCallOpKernelPhyInstrOperand() override = default;

  template<typename... Args>
  static Maybe<LocalCallOpKernelPhyInstrOperand> New(Args&&... args) {
    auto* ptr = new LocalCallOpKernelPhyInstrOperand(std::forward<Args>(args)...);
    JUST(ptr->Init());
    return std::shared_ptr<LocalCallOpKernelPhyInstrOperand>(ptr);
  }

  const one::StatefulLocalOpKernel& opkernel() const { return *opkernel_; }
  const std::shared_ptr<one::StatefulLocalOpKernel>& shared_opkernel() const { return opkernel_; }
  const one::EagerBlobObjectListPtr& inputs() const { return inputs_; }
  const one::EagerBlobObjectListPtr& outputs() const { return outputs_; }
  const AttrMap& attrs() const { return op_interp_ctx_.attrs; }
  const one::OpExprInterpContext& op_interp_ctx() const { return op_interp_ctx_; }
  const one::DevVmDepObjectConsumeMode& dev_vm_dep_object_consume_mode() const {
    return dev_vm_dep_object_consume_mode_;
  }

  one::StatefulLocalOpKernel* mut_opkernel() { return opkernel_.get(); }

  template<typename DoEachT>
  Maybe<void> ForEachOutputTensor(const DoEachT& DoEach) {
    for (const auto& output : *outputs()) { JUST(DoEach(output.get())); }
    return Maybe<void>::Ok();
  }

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  void ForEachConstMirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const;

  void ForEachMutMirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const;

  void ForEachMut2MirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const;

  bool need_temp_storage() const { return need_temp_storage_; }
  const user_op::OpKernel* user_opkernel() const { return user_opkernel_; }

  const std::shared_ptr<const one::ConsistentTensorInferResult>& consistent_tensor_infer_result()
      const {
    return consistent_tensor_infer_result_;
  }

 private:
  LocalCallOpKernelPhyInstrOperand(
      const std::shared_ptr<one::StatefulLocalOpKernel>& opkernel,
      const one::EagerBlobObjectListPtr& inputs, const one::EagerBlobObjectListPtr& outputs,
      const std::shared_ptr<const one::ConsistentTensorInferResult>& consistent_tensor_infer_result,
      const one::OpExprInterpContext& op_interp_ctx_,
      const one::DevVmDepObjectConsumeMode dev_vm_dep_object_consume_mode)
      : opkernel_(opkernel),
        inputs_(inputs),
        outputs_(outputs),
        consistent_tensor_infer_result_(consistent_tensor_infer_result),
        op_interp_ctx_(op_interp_ctx_),
        dev_vm_dep_object_consume_mode_(dev_vm_dep_object_consume_mode),
        input_dependences_(),
        output_dependences_() {
    ForEachConstMirroredObject(SetInserter(&input_dependences_));
    ForEachMutMirroredObject(SetInserter(&output_dependences_));
    ForEachMut2MirroredObject(SetInserter(&output_dependences_));
    InitStreamSequentialDependence();
  }

  Maybe<void> Init();
  void InitStreamSequentialDependence();

  std::shared_ptr<one::StatefulLocalOpKernel> opkernel_;
  one::EagerBlobObjectListPtr inputs_;
  one::EagerBlobObjectListPtr outputs_;
  std::shared_ptr<const one::ConsistentTensorInferResult> consistent_tensor_infer_result_;
  const one::OpExprInterpContext op_interp_ctx_;
  const user_op::OpKernel* user_opkernel_;
  bool need_temp_storage_;
  const one::DevVmDepObjectConsumeMode dev_vm_dep_object_consume_mode_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

class DTRInstrOperand {
 public:
  DTRInstrOperand(const DTRInstrOperand&) = delete;
  DTRInstrOperand(DTRInstrOperand&&) = delete;

  DTRInstrOperand(
      const std::shared_ptr<one::StatefulLocalOpKernel>& opkernel,
      const one::EagerBlobObjectListPtr& input, const one::EagerBlobObjectListPtr& output,
      const std::shared_ptr<const one::ConsistentTensorInferResult>& consistent_tensor_infer_result,
      const one::OpExprInterpContext& op_interp_ctx_,
      const one::DevVmDepObjectConsumeMode dev_vm_dep_object_consume_mode);
  ~DTRInstrOperand() = default;

  const std::shared_ptr<one::StatefulLocalOpKernel>& shared_opkernel() const { return opkernel_; }
  const std::vector<std::weak_ptr<vm::EagerBlobObject>>& inputs() const { return inputs_; }
  const std::vector<std::weak_ptr<vm::EagerBlobObject>>& outputs() const { return outputs_; }
  const one::OpExprInterpContext& op_interp_ctx() const { return op_interp_ctx_; }
  const one::DevVmDepObjectConsumeMode& dev_vm_dep_object_consume_mode() const {
    return dev_vm_dep_object_consume_mode_;
  }
  const std::shared_ptr<const one::ConsistentTensorInferResult>& consistent_tensor_infer_result()
      const {
    return consistent_tensor_infer_result_;
  }

 private:
  std::shared_ptr<one::StatefulLocalOpKernel> opkernel_;
  std::vector<std::weak_ptr<vm::EagerBlobObject>> inputs_;
  std::vector<std::weak_ptr<vm::EagerBlobObject>> outputs_;
  std::shared_ptr<const one::ConsistentTensorInferResult> consistent_tensor_infer_result_;
  const one::OpExprInterpContext op_interp_ctx_;
  const one::DevVmDepObjectConsumeMode dev_vm_dep_object_consume_mode_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_
