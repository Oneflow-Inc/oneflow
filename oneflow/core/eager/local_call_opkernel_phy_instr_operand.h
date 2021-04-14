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

using TensorsPtr = std::shared_ptr<std::vector<std::shared_ptr<eager::EagerBlobObject>>>;
using TensorIndexMap = std::vector<std::pair<std::string, int>>;

}  // namespace one

namespace user_op {

class InferContext;

}

namespace eager {

class LocalCallOpKernelPhyInstrOperand final : public vm::PhyInstrOperand {
 public:
  LocalCallOpKernelPhyInstrOperand(const LocalCallOpKernelPhyInstrOperand&) = delete;
  LocalCallOpKernelPhyInstrOperand(LocalCallOpKernelPhyInstrOperand&&) = delete;
  ~LocalCallOpKernelPhyInstrOperand() override = default;

  LocalCallOpKernelPhyInstrOperand(const std::shared_ptr<one::StatefulOpKernel>& opkernel,
                                   const one::TensorsPtr inputs, const one::TensorsPtr outputs)
      : opkernel_(opkernel), inputs_(inputs), outputs_(outputs) {}

  const one::StatefulOpKernel& opkernel() const { return *opkernel_; }
  const one::TensorsPtr inputs() const { return inputs_; }
  const one::TensorsPtr outputs() const { return outputs_; }

  one::StatefulOpKernel* mut_opkernel() { return opkernel_.get(); }
  const one::TensorsPtr mut_inputs() { return inputs_; }
  const one::TensorsPtr mut_outputs() { return outputs_; }

  using OutputFn = std::function<Maybe<void>(eager::EagerBlobObject* tensor)>;

  Maybe<void> ForEachOutputTensor(OutputFn func) {
    for (auto& output : *mut_outputs()) { JUST(func(output.get())); }
    return Maybe<void>::Ok();
  }

  void ForEachInferMutMirroredObject(
      const std::function<void(vm::MirroredObject*)>&) const override {
    // do nothing
  }
  void ForEachInferConstMirroredObject(
      const std::function<void(vm::MirroredObject*)>&) const override {
    // do nothing
  }
  void ForEachComputeMutMirroredObject(
      const std::function<void(vm::MirroredObject*)>&) const override {
    // do nothing
  }
  void ForEachComputeConstMirroredObject(
      const std::function<void(vm::MirroredObject*)>&) const override {
    // do nothing
  }

 private:
  std::shared_ptr<one::StatefulOpKernel> opkernel_;
  one::TensorsPtr inputs_;
  one::TensorsPtr outputs_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_
