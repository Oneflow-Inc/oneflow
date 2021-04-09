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
#ifndef ONEFLOW_CORE_VM_NO_ARG_CB_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_VM_NO_ARG_CB_PHY_INSTR_OPERAND_H_

#include <functional>
#include "oneflow/core/vm/phy_instr_operand.h"

namespace oneflow {
namespace vm {

// no arg callback physical instruction operand
class NoArgCbPhyInstrOperand : public PhyInstrOperand {
 public:
  NoArgCbPhyInstrOperand(const std::function<void()>& callback) : callback_(callback) {}
  ~NoArgCbPhyInstrOperand() = default;

  const std::function<void()>& callback() const { return callback_; }

  void ForEachInferMutMirroredObject(const std::function<void(MirroredObject*)>&) const override {
    // do nothing
  }
  void ForEachInferConstMirroredObject(const std::function<void(MirroredObject*)>&) const override {
    // do nothing
  }
  void ForEachComputeMutMirroredObject(const std::function<void(MirroredObject*)>&) const override {
    // do nothing
  }
  void ForEachComputeConstMirroredObject(
      const std::function<void(MirroredObject*)>&) const override {
    // do nothing
  }

 private:
  std::function<void()> callback_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_NO_ARG_CB_PHY_INSTR_OPERAND_H_
