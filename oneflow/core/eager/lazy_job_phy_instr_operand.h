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

  LaunchLazyJobPhyInstrOperand(
      const intrusive::shared_ptr<LocalDepObject>& inputs_local_dep_object,
      const intrusive::shared_ptr<LocalDepObject>& outputs_local_dep_object,
      const std::shared_ptr<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>&
          op_name2end_event_record,
      const one::EagerBlobObjectListPtr& input_blob_objects,
      const one::EagerBlobObjectListPtr& output_blob_objects,
      const one::EagerBlobObjectListPtr& param_blob_objects,
      const std::shared_ptr<NNGraphIf>& nn_graph)
      : inputs_local_dep_object_(inputs_local_dep_object),
        outputs_local_dep_object_(outputs_local_dep_object),
        op_name2end_event_record_(op_name2end_event_record),
        input_blob_objects_(input_blob_objects),
        output_blob_objects_(output_blob_objects),
        param_blob_objects_(param_blob_objects),
        nn_graph_(nn_graph) {}

  const one::EagerBlobObjectListPtr& input_blob_objects() const { return input_blob_objects_; }
  const one::EagerBlobObjectListPtr& output_blob_objects() const { return output_blob_objects_; }
  const std::shared_ptr<NNGraphIf>& nn_graph() const { return nn_graph_; }

  Maybe<SharedEventRecord> EndEventRecord4OpName(const std::string& op_name) const;
  const std::shared_ptr<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>&
  op_name2end_event_record() const {
    return op_name2end_event_record_;
  }

  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* compute)>&) const override;

  void ForEachMutMirroredObject(
      const std::function<void(vm::MirroredObject* compute)>&) const override;

  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* compute)>&) const override;

 private:
  mutable intrusive::shared_ptr<LocalDepObject> inputs_local_dep_object_;
  mutable intrusive::shared_ptr<LocalDepObject> outputs_local_dep_object_;
  std::shared_ptr<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>
      op_name2end_event_record_;
  one::EagerBlobObjectListPtr input_blob_objects_;
  one::EagerBlobObjectListPtr output_blob_objects_;
  one::EagerBlobObjectListPtr param_blob_objects_;
  std::shared_ptr<NNGraphIf> nn_graph_;
};
}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LAZY_JOB_PHY_INSTR_OPERAND_H_
