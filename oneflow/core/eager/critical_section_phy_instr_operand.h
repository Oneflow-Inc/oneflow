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
#ifndef ONEFLOW_CORE_EAGER_CRITICAL_SECTION_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_CRITICAL_SECTION_PHY_INSTR_OPERAND_H_

#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/nn_graph_if.h"

namespace oneflow {

namespace one {

using EagerBlobObjectListPtr =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;

}

namespace vm {

class CriticalSectionPhyInstrOperand : public PhyInstrOperand {
 public:
  CriticalSectionPhyInstrOperand(const CriticalSectionPhyInstrOperand&) = delete;
  CriticalSectionPhyInstrOperand(CriticalSectionPhyInstrOperand&&) = delete;
  virtual ~CriticalSectionPhyInstrOperand() = default;

  explicit CriticalSectionPhyInstrOperand(int64_t ref_cnt);

  const std::shared_ptr<std::atomic<int64_t>>& critical_section_ready_ref_cnt() const {
    return critical_section_ready_ref_cnt_;
  }

  const std::shared_ptr<std::atomic<int64_t>>& consumer_ref_cnt() const {
    return consumer_ref_cnt_;
  }

 protected:
  // Initiliazed with 1.
  // Reset to 0 when critical section ready.
  std::shared_ptr<std::atomic<int64_t>> critical_section_ready_ref_cnt_;
  // Number of working consumers.
  std::shared_ptr<std::atomic<int64_t>> consumer_ref_cnt_;
};

class TensorCriticalSectionPhyInstrOperand : public CriticalSectionPhyInstrOperand {
 public:
  TensorCriticalSectionPhyInstrOperand(const TensorCriticalSectionPhyInstrOperand&) = delete;
  TensorCriticalSectionPhyInstrOperand(TensorCriticalSectionPhyInstrOperand&&) = delete;
  virtual ~TensorCriticalSectionPhyInstrOperand() = default;

  TensorCriticalSectionPhyInstrOperand(const one::EagerBlobObjectListPtr& eager_blob_objects,
                                       const HashMap<std::string, int64_t>& op_name2index,
                                       const std::shared_ptr<NNGraphIf>& nn_graph);

  const std::shared_ptr<NNGraphIf>& nn_graph() const { return nn_graph_; }

  // Called by consumer.
  void ConsumerFetchBlobAndDecreaseRefCnt(const std::string& op_name,
                                          const std::function<void(Blob*)>& Callback) const;

  void ForEachMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&) const;

  const std::shared_ptr<std::atomic<int64_t>>& consumer_ref_cnt() const {
    return consumer_ref_cnt_;
  };
  const HashMap<std::string, int64_t>& op_name2index() const { return op_name2index_; }

 protected:
  one::EagerBlobObjectListPtr eager_blob_objects_;
  std::shared_ptr<NNGraphIf> nn_graph_;
  // op_name to index within `eager_blob_objects_`.
  HashMap<std::string, int64_t> op_name2index_;
};

class InputCriticalSectionPhyInstrOperand final : public TensorCriticalSectionPhyInstrOperand {
 public:
  InputCriticalSectionPhyInstrOperand(const one::EagerBlobObjectListPtr& eager_blob_objects,
                                      const std::shared_ptr<NNGraphIf>& nn_graph)
      : TensorCriticalSectionPhyInstrOperand(eager_blob_objects, GetOpNames(*nn_graph), nn_graph) {}

  ~InputCriticalSectionPhyInstrOperand() override = default;

  // for inputs
  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
      const override {
    ForEachMirroredObject(DoEach);
  }

  void ForEachMutMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  // for outputs
  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override {}

 private:
  HashMap<std::string, int64_t> GetOpNames(const NNGraphIf& nn_graph) const {
    HashMap<std::string, int64_t> ret;
    const auto& op_names = nn_graph.inputs_op_names();
    for (int i = 0; i < op_names.size(); ++i) { CHECK(ret.emplace(op_names.at(i), i).second); }
    return ret;
  }
};

class OutputCriticalSectionPhyInstrOperand final : public TensorCriticalSectionPhyInstrOperand {
 public:
  OutputCriticalSectionPhyInstrOperand(const one::EagerBlobObjectListPtr& eager_blob_objects,
                                       const std::shared_ptr<NNGraphIf>& nn_graph)
      : TensorCriticalSectionPhyInstrOperand(eager_blob_objects, GetOpNames(*nn_graph), nn_graph) {}

  ~OutputCriticalSectionPhyInstrOperand() override = default;

  // for inputs
  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override {}

  // for params
  void ForEachMutMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  // for outputs
  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
      const override {
    ForEachMirroredObject(DoEach);
  }

 private:
  HashMap<std::string, int64_t> GetOpNames(const NNGraphIf& nn_graph) const {
    HashMap<std::string, int64_t> ret;
    const auto& op_names = nn_graph.outputs_op_names();
    for (int i = 0; i < op_names.size(); ++i) { CHECK(ret.emplace(op_names.at(i), i).second); }
    return ret;
  }
};

class ParameterCriticalSectionPhyInstrOperand final : public TensorCriticalSectionPhyInstrOperand {
 public:
  ParameterCriticalSectionPhyInstrOperand(const one::EagerBlobObjectListPtr& eager_blob_objects,
                                          const std::shared_ptr<NNGraphIf>& nn_graph)
      : TensorCriticalSectionPhyInstrOperand(eager_blob_objects, {}, nn_graph) {}

  ~ParameterCriticalSectionPhyInstrOperand() override = default;

  // for inputs
  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override {}

  // for params
  void ForEachMutMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
      const override;

  // for outputs
  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override {}
};

class NcclCriticalSectionPhyInstrOperand final : public CriticalSectionPhyInstrOperand {
 public:
  NcclCriticalSectionPhyInstrOperand() : CriticalSectionPhyInstrOperand(1) {}

  ~NcclCriticalSectionPhyInstrOperand() override = default;

  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override {}

  void ForEachMutMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
      const override;

  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override {}
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_CRITICAL_SECTION_PHY_INSTR_OPERAND_H_
