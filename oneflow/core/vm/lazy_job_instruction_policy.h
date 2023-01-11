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
#ifndef ONEFLOW_CORE_VM_LAZY_JOB_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_LAZY_JOB_INSTRUCTION_POLICY_H_

#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/of_unused.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/vm/instruction_policy.h"
#include "oneflow/core/vm/instruction_policy_util.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/vm/lazy_job_stream_policy.h"
#include "oneflow/core/vm/virtual_machine.h"
#include <robin_hood.h>

namespace oneflow {

class LazyJobInstance final : public JobInstance {
 public:
  LazyJobInstance(const LazyJobInstance&) = delete;
  LazyJobInstance(LazyJobInstance&&) = delete;
  ~LazyJobInstance() override = default;
  LazyJobInstance(const std::string& job_name, const std::function<void()>& finish_cb)
      : job_name_(job_name), finish_cb_(finish_cb) {}

  std::string job_name() const override { return job_name_; }
  void Finish() const override { finish_cb_(); }

 private:
  const std::string job_name_;
  const std::function<void()> finish_cb_;
};

namespace vm {

class LaunchLazyJobInstructionPolicy final : public InstructionPolicy {  // NOLINT
 public:
  LaunchLazyJobInstructionPolicy(const LaunchLazyJobInstructionPolicy&) = delete;
  LaunchLazyJobInstructionPolicy(LaunchLazyJobInstructionPolicy&&) = delete;
  ~LaunchLazyJobInstructionPolicy() = default;

  LaunchLazyJobInstructionPolicy(const std::shared_ptr<NNGraphIf>& nn_graph,
                                 const EagerBlobObjectListPtr& param_blob_objects)
      : nn_graph_(nn_graph),
        param_blob_objects_(param_blob_objects),
        input_dependences_(),
        output_dependences_() {
    robin_hood::unordered_flat_map<Dependence*, bool> unique_map;
    ForEachConstDependence([&](Dependence* compute) {
      if (unique_map.emplace(compute, true).second) { input_dependences_.emplace_back(compute); }
    });
    unique_map.clear();
    output_dependences_.reserve(param_blob_objects_->size());
    unique_map.reserve(param_blob_objects_->size());
    ForEachMutDependence([&](Dependence* compute) {
      if (unique_map.emplace(compute, true).second) { output_dependences_.emplace_back(compute); }
    });
    ForEachMut2Dependence([&](Dependence* compute) {
      if (unique_map.emplace(compute, true).second) { output_dependences_.emplace_back(compute); }
    });
  }

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  void ForEachConstDependence(const std::function<void(Dependence* compute)>&) const {}

  void ForEachMutDependence(const std::function<void(Dependence* compute)>& DoEach) const {
    for (const auto& eager_blob_object : *param_blob_objects_) {
      DoEach(CHECK_JUST(eager_blob_object->compute_local_dep_object()));
    }
    DoEach(CHECK_JUST(SingletonMaybe<VirtualMachine>())
               ->FindOrCreateTransportLocalDepObject()
               .Mutable());
  }

  void ForEachMut2Dependence(const std::function<void(Dependence* compute)>&) const {}

  std::string DebugName(const Instruction&) const override { return "LaunchLazyJob"; }
  Maybe<void> Prepare(Instruction* instruction) override { return Maybe<void>::Ok(); }
  void Compute(Instruction* instruction) override {
    auto* lazy_job_stream_policy = GetLazyJobStreamPolicy(instruction);

    static thread_local int64_t run_id = 0;
    {
      OF_PROFILER_RANGE_GUARD("WaitUntilQueueEmptyIfFrontNNGraphNotEquals");
      lazy_job_stream_policy->WaitUntilQueueEmptyIfFrontNNGraphNotEquals(nn_graph_);
    }
    {
      OF_PROFILER_RANGE_GUARD("Send all buffers to BufferMgr");
      const auto& job_instance = MakeJobInstance(instruction);
      const auto& job_name = job_instance->job_name();
      auto* buffer_mgr = Singleton<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
      buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Push(job_instance);
      buffer_mgr->Get(GetSourceTickBufferName(job_name))->Push(job_instance);
    }
    OF_UNUSED(run_id);  // disable compiler warning.
    OF_PROFILER_RANGE_GUARD("EnqueueNNGraph");
    lazy_job_stream_policy->EnqueueNNGraph(nn_graph_);
  }

 private:
  LazyJobStreamPolicy* GetLazyJobStreamPolicy(Instruction* instruction) const {
    StreamPolicy* stream_policy = instruction->mut_stream()->mut_stream_policy();
    LazyJobStreamPolicy* lazy_job_stream_policy = dynamic_cast<LazyJobStreamPolicy*>(stream_policy);
    CHECK_NOTNULL(lazy_job_stream_policy);
    return lazy_job_stream_policy;
  }

  std::shared_ptr<LazyJobInstance> MakeJobInstance(Instruction* instruction) const {
    const auto& FinishCb = [this, instruction]() {
      auto* lazy_job_stream_policy = GetLazyJobStreamPolicy(instruction);
      lazy_job_stream_policy->DequeueNNGraph();
      auto* status_buffer = instruction->mut_status_buffer();
      NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer())->set_done();
    };
    return std::make_shared<LazyJobInstance>(nn_graph_->job_name(), FinishCb);
  }

  std::shared_ptr<NNGraphIf> nn_graph_;
  EagerBlobObjectListPtr param_blob_objects_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow
#endif  // ONEFLOW_CORE_VM_LAZY_JOB_INSTRUCTION_POLICY_H_
