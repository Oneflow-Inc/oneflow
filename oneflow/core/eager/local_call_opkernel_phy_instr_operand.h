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

#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/instruction_operand.msg.h"

namespace oneflow {

namespace one {

class TensorTuple;
class StatefulOpKernel;

using EagerBlobObjectList = std::shared_ptr<std::vector<std::shared_ptr<eager::EagerBlobObject>>>;
using TensorIndexMap = std::vector<std::pair<std::string, int>>;

}  // namespace one

namespace user_op {

class InferContext;
class OpKernel;

}  // namespace user_op

namespace eager {

class LocalCallOpKernelPhyInstrOperand final : public vm::PhyInstrOperand {
 public:
  LocalCallOpKernelPhyInstrOperand(const LocalCallOpKernelPhyInstrOperand&) = delete;
  LocalCallOpKernelPhyInstrOperand(LocalCallOpKernelPhyInstrOperand&&) = delete;
  ~LocalCallOpKernelPhyInstrOperand() override = default;

  LocalCallOpKernelPhyInstrOperand(const std::shared_ptr<one::StatefulOpKernel>& opkernel,
                                   const one::EagerBlobObjectList inputs,
                                   const one::EagerBlobObjectList outputs)
      : opkernel_(opkernel), inputs_(inputs), outputs_(outputs) {}

  const one::StatefulOpKernel& opkernel() const { return *opkernel_; }
  const one::EagerBlobObjectList inputs() const { return inputs_; }
  const one::EagerBlobObjectList outputs() const { return outputs_; }

  one::StatefulOpKernel* mut_opkernel() { return opkernel_.get(); }
  const one::EagerBlobObjectList mut_inputs() { return inputs_; }
  const one::EagerBlobObjectList mut_outputs() { return outputs_; }

  using OutputFn = std::function<Maybe<void>(eager::EagerBlobObject* tensor)>;

  Maybe<void> ForEachOutputTensor(OutputFn func) {
    for (auto& output : *mut_outputs()) { JUST(func(output.get())); }
    return Maybe<void>::Ok();
  }

  void ForEachInferMutMirroredObject(
      const std::function<void(vm::MirroredObject*)>& fn) const override;
  void ForEachInferConstMirroredObject(
      const std::function<void(vm::MirroredObject*)>&) const override {
    // TODO:
  }
  void ForEachComputeMutMirroredObject(
      const std::function<void(vm::MirroredObject*)>&) const override {
    // TODO:
  }
  void ForEachComputeConstMirroredObject(
      const std::function<void(vm::MirroredObject*)>&) const override {
    // TODO:
  }

  const user_op::OpKernel* user_opkernel() const { return user_opkernel_; }

  void set_user_opkernel(const user_op::OpKernel* user_opkernel) { user_opkernel_ = user_opkernel; }

 private:
  std::shared_ptr<one::StatefulOpKernel> opkernel_;
  one::EagerBlobObjectList inputs_;
  one::EagerBlobObjectList outputs_;
  const user_op::OpKernel* user_opkernel_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_
