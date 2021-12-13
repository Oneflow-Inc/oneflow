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
#ifndef ONEFLOW_CORE_EAGER_ALLOCATE_OUTPUTS_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_ALLOCATE_OUTPUTS_PHY_INSTR_OPERAND_H_

#include "oneflow/core/eager/dev_vm_dep_object_consume_mode.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/vm/instruction_operand.h"

namespace oneflow {
namespace one {

using EagerBlobObjectList = std::vector<std::shared_ptr<vm::EagerBlobObject>>;
using EagerBlobObjectListPtr =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;

}  // namespace one

namespace vm {

class AllocateOutputsPhyInstrOperand final : public vm::PhyInstrOperand {
 public:
  AllocateOutputsPhyInstrOperand(const AllocateOutputsPhyInstrOperand&) = delete;
  AllocateOutputsPhyInstrOperand(AllocateOutputsPhyInstrOperand&&) = delete;
  ~AllocateOutputsPhyInstrOperand() override = default;

  explicit AllocateOutputsPhyInstrOperand(const one::EagerBlobObjectListPtr& outputs)
      : outputs_(outputs), output_dependences_() {
    for (const auto& output : *outputs_) {
      output_dependences_.push_back(
          CHECK_JUST(output->compute_local_dep_object())->mut_mirrored_object());
    }
  }

  const one::EagerBlobObjectListPtr& outputs() const { return outputs_; }

  const DependenceVector& input_dependences() const override {
    static DependenceVector empty{};
    return empty;
  }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

 private:
  one::EagerBlobObjectListPtr outputs_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_ALLOCATE_OUTPUTS_PHY_INSTR_OPERAND_H_
