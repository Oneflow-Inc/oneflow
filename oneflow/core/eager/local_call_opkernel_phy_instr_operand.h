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

#include "oneflow/core/eager/dev_vm_dep_object_consume_mode.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/vm/instruction_operand.msg.h"

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
        dev_vm_dep_object_consume_mode_(dev_vm_dep_object_consume_mode) {}

  const one::StatefulLocalOpKernel& opkernel() const { return *opkernel_; }
  const std::shared_ptr<one::StatefulLocalOpKernel>& shared_opkernel() const {return opkernel_; }
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

  template<typename DoEachT>
  Maybe<void> ForEachInputTensor(const DoEachT& DoEach) {
    for (const auto& input : *inputs()) { JUST(DoEach(input.get())); }
    return Maybe<void>::Ok();
  }

  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  void ForEachMutMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  const user_op::OpKernel* user_opkernel() const { return user_opkernel_; }

  void set_user_opkernel(const user_op::OpKernel* user_opkernel) { user_opkernel_ = user_opkernel; }

  const std::shared_ptr<const one::ConsistentTensorInferResult>& consistent_tensor_infer_result()
      const {
    return consistent_tensor_infer_result_;
  }

 protected:
  std::shared_ptr<one::StatefulLocalOpKernel> opkernel_;
  one::EagerBlobObjectListPtr inputs_;
  one::EagerBlobObjectListPtr outputs_;
  std::shared_ptr<const one::ConsistentTensorInferResult> consistent_tensor_infer_result_;
  const one::OpExprInterpContext op_interp_ctx_;
  const user_op::OpKernel* user_opkernel_;
  const one::DevVmDepObjectConsumeMode dev_vm_dep_object_consume_mode_;
};


class DTRLocalCallOpKernelPhyInstrOperand final : public LocalCallOpKernelPhyInstrOperand {
 public:
  DTRLocalCallOpKernelPhyInstrOperand(const DTRLocalCallOpKernelPhyInstrOperand&) = delete;
  DTRLocalCallOpKernelPhyInstrOperand(DTRLocalCallOpKernelPhyInstrOperand&&) = delete;

  DTRLocalCallOpKernelPhyInstrOperand(
      const std::shared_ptr<one::StatefulLocalOpKernel>& opkernel,
      const one::EagerBlobObjectListPtr& inputs, const one::EagerBlobObjectListPtr& outputs,
      const std::shared_ptr<const one::ConsistentTensorInferResult>& consistent_tensor_infer_result,
      const one::OpExprInterpContext& op_interp_ctx_,
      const one::DevVmDepObjectConsumeMode dev_vm_dep_object_consume_mode) : LocalCallOpKernelPhyInstrOperand(opkernel, inputs, outputs, consistent_tensor_infer_result, op_interp_ctx_, dev_vm_dep_object_consume_mode)
      {
        std::shared_ptr<one::EagerBlobObjectList> tmp_inputs = std::make_shared<one::EagerBlobObjectList>(inputs->size());
        for (int i = 0; i < inputs->size(); ++i) {
          tmp_inputs->at(i) = std::shared_ptr<vm::EagerBlobObject>(inputs_->at(i).get(), [](vm::EagerBlobObject* ptr) {});
          // tmp_inputs->at(i) = std::shared_ptr<vm::EagerBlobObject>(inputs_->at(i).get(), [](vm::EagerBlobObject* ptr) { std::cout << "Fake delete inputs in the copied operand." << std::endl; });
        }
        inputs_ = tmp_inputs;
        std::shared_ptr<one::EagerBlobObjectList> tmp_outputs = std::make_shared<one::EagerBlobObjectList>(outputs->size());
        for (int i = 0; i < outputs->size(); ++i) {
          tmp_outputs->at(i) = std::shared_ptr<vm::EagerBlobObject>(outputs_->at(i).get(), [](vm::EagerBlobObject* ptr) {});
          // tmp_outputs->at(i) = std::shared_ptr<vm::EagerBlobObject>(outputs_->at(i).get(), [](vm::EagerBlobObject* ptr) { std::cout << "Fake delete outputs in the copied operand." << std::endl; });
        }
        outputs_ = tmp_outputs;
      }
  ~DTRLocalCallOpKernelPhyInstrOperand() override = default;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_