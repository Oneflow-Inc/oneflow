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
#ifndef ONEFLOW_CORE_EAGER_LAZY_JOB_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_LAZY_JOB_PHY_INSTR_OPERAND_H_

#include "oneflow/core/vm/instruction_operand.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/device/event_record.h"
#include "oneflow/core/eager/critical_section_phy_instr_operand.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/notifier.h"

namespace oneflow {

namespace one {

using EagerBlobObjectListPtr =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;

}

namespace vm {

class LaunchLazyJobPhyInstrOperand final : public PhyInstrOperand {
 public:
  LaunchLazyJobPhyInstrOperand(const LaunchLazyJobPhyInstrOperand&) = delete;
  LaunchLazyJobPhyInstrOperand(LaunchLazyJobPhyInstrOperand&&) = delete;
  ~LaunchLazyJobPhyInstrOperand() override = default;

  LaunchLazyJobPhyInstrOperand(const std::shared_ptr<NNGraphIf>& nn_graph,
                               const one::EagerBlobObjectListPtr& param_blob_objects)
      : nn_graph_(nn_graph),
        param_blob_objects_(param_blob_objects),
        input_dependences_(),
        output_dependences_() {
    ForEachConstMirroredObject(SetInserter(&input_dependences_));
    ForEachMutMirroredObject(SetInserter(&output_dependences_));
    ForEachMut2MirroredObject(SetInserter(&output_dependences_));
    stream_sequential_dependence_ = nullptr;
  }

  const std::shared_ptr<NNGraphIf>& nn_graph() const { return nn_graph_; }

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  void ForEachConstMirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const {}

  void ForEachMutMirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const;

  void ForEachMut2MirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const {}

 private:
  std::shared_ptr<NNGraphIf> nn_graph_;
  one::EagerBlobObjectListPtr param_blob_objects_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};
}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LAZY_JOB_PHY_INSTR_OPERAND_H_
