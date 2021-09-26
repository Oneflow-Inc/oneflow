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

#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/local_dep_object.h"
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

  LaunchLazyJobPhyInstrOperand(
      const std::vector<ObjectMsgPtr<LocalDepObject>>& input_local_dep_objects,
      const std::vector<ObjectMsgPtr<LocalDepObject>>& output_local_dep_objects,
      HashMap<std::string, >
      const one::EagerBlobObjectListPtr& param_blob_objects,
      const std::shared_ptr<NNGraphIf>& nn_graph)
      : input_local_dep_objects_(input_local_dep_objects),
        output_local_dep_objects_(output_local_dep_objects),
        param_blob_objects_(param_blob_objects),
        nn_graph_(nn_graph) {}

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
  std::vector<ObjectMsgPtr<LocalDepObject>> input_local_dep_objects_;
  std::vector<ObjectMsgPtr<LocalDepObject>> output_local_dep_objects_;
  one::EagerBlobObjectListPtr param_blob_objects_;
  std::shared_ptr<NNGraphIf> nn_graph_;
};
}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LAZY_JOB_PHY_INSTR_OPERAND_H_
