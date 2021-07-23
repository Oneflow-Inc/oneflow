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
#ifndef ONEFLOW_CORE_EAGER_RUN_JOB_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_RUN_JOB_PHY_INSTR_OPERAND_H_

#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/nn_graph_if.h"

namespace oneflow {

namespace one {

using EagerBlobObjectListPtr =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;

}

namespace vm {

class RunLazyJobPhyInstrOperand final : public PhyInstrOperand {
 public:
  RunLazyJobPhyInstrOperand(const RunLazyJobPhyInstrOperand&) = delete;
  RunLazyJobPhyInstrOperand(RunLazyJobPhyInstrOperand&&) = delete;
  ~RunLazyJobPhyInstrOperand() override = default;

  RunLazyJobPhyInstrOperand(const one::EagerBlobObjectListPtr& inputs,
                            const one::EagerBlobObjectListPtr& outputs,
                            const one::EagerBlobObjectListPtr& parameters,
                            const std::shared_ptr<NNGraphIf>& nn_graph)
      : inputs_(inputs), outputs_(outputs), parameters_(parameters), nn_graph_(nn_graph) {}

  const one::EagerBlobObjectListPtr& inputs() const { return inputs_; }
  const one::EagerBlobObjectListPtr& outputs() const { return outputs_; }
  const one::EagerBlobObjectListPtr& parameters() const { return parameters_; }
  const std::shared_ptr<NNGraphIf>& nn_graph() const { return nn_graph_; }

  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  void ForEachMutMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

 private:
  one::EagerBlobObjectListPtr inputs_;
  one::EagerBlobObjectListPtr outputs_;
  one::EagerBlobObjectListPtr parameters_;
  std::shared_ptr<NNGraphIf> nn_graph_;
};
}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_RUN_JOB_PHY_INSTR_OPERAND_H_
