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
#ifndef ONEFLOW_CORE_EAGER_WAIT_UNTIL_ZERO_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_WAIT_UNTIL_ZERO_PHY_INSTR_OPERAND_H_

#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/local_dep_object.h"

namespace oneflow {
namespace vm {

class WaitUntilZeroPhyInstrOperand : public PhyInstrOperand {
 public:
  WaitUntilZeroPhyInstrOperand(const WaitUntilZeroPhyInstrOperand&) = delete;
  WaitUntilZeroPhyInstrOperand(WaitUntilZeroPhyInstrOperand&&) = delete;
  virtual ~WaitUntilZeroPhyInstrOperand() = default;

  explicit WaitUntilZeroPhyInstrOperand(std::shared_ptr<std::atomic<int64_t>> ref_cnt,
                                        ObjectMsgPtr<LocalDepObject> local_dep_object)
      : ref_cnt_(ref_cnt), local_dep_object_(local_dep_object) {}

  const std::shared_ptr<std::atomic<int64_t>>& ref_cnt() const { return ref_cnt_; }

  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override {}

  void ForEachMutMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
      const override {
    DoEach(nullptr, local_dep_object_->mut_mirrored_object());
  }

  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override {}

 private:
  std::shared_ptr<std::atomic<int64_t>> ref_cnt_;
  mutable ObjectMsgPtr<LocalDepObject> local_dep_object_;
};

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_WAIT_UNTIL_ZERO_PHY_INSTR_OPERAND_H_
