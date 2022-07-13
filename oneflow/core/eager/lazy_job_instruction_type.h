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
#ifndef ONEFLOW_CORE_EAGER_LAZY_JOB_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_EAGER_LAZY_JOB_INSTRUCTION_TYPE_H_

#include "oneflow/core/vm/lazy_job_device_context.h"
#include "oneflow/core/eager/lazy_job_phy_instr_operand.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/of_unused.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/naive_stream_policy.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/kernel/kernel_util.h"

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

  std::string sole_input_op_name_in_user_job() const override {
    UNIMPLEMENTED();
    return std::string();
  }
  std::string sole_output_op_name_in_user_job() const override {
    UNIMPLEMENTED();
    return std::string();
  }
  void PushBlob(uint64_t ofblob_ptr) const override { UNIMPLEMENTED(); }
  void PullBlob(uint64_t ofblob_ptr) const override { UNIMPLEMENTED(); }

 private:
  const std::string job_name_;
  const std::function<void()> finish_cb_;
};

namespace vm {

class LaunchLazyJobInstructionType final : public InstructionType {  // NOLINT
 public:
  LaunchLazyJobInstructionType(const LaunchLazyJobInstructionType&) = delete;
  LaunchLazyJobInstructionType(LaunchLazyJobInstructionType&&) = delete;
  LaunchLazyJobInstructionType() = default;
  ~LaunchLazyJobInstructionType() = default;

  std::string DebugName(const vm::Instruction&) const override { return "LaunchLazyJob"; }
  Maybe<void> Prepare(vm::Instruction* instruction) const override { return Maybe<void>::Ok(); }
  void Compute(vm::Instruction* instruction) const override {
    const auto& cur_nn_graph = GetCurNNGraph(instruction);
    auto* device_ctx = GetLazyJobDeviceCtx(instruction);

    static thread_local int64_t run_id = 0;
    {
      OF_PROFILER_RANGE_GUARD("WaitUntilQueueEmptyIfFrontNNGraphNotEquals");
      device_ctx->WaitUntilQueueEmptyIfFrontNNGraphNotEquals(cur_nn_graph);
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
    device_ctx->EnqueueNNGraph(cur_nn_graph);
  }

 private:
  LazyJobDeviceCtx* GetLazyJobDeviceCtx(Instruction* instruction) const {
    StreamPolicy* stream_policy = instruction->mut_stream()->mut_stream_policy();
    NaiveStreamPolicy* naive_stream_policy = dynamic_cast<NaiveStreamPolicy*>(stream_policy);
    CHECK_NOTNULL(naive_stream_policy);
    auto* device_ctx = dynamic_cast<LazyJobDeviceCtx*>(naive_stream_policy->device_ctx().get());
    CHECK_NOTNULL(device_ctx);
    return device_ctx;
  }
  std::shared_ptr<NNGraphIf> GetCurNNGraph(Instruction* instruction) const {
    const auto* ptr = instruction->phy_instr_operand().get();
    const auto* phy_instr_operand = dynamic_cast<const LaunchLazyJobPhyInstrOperand*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    return phy_instr_operand->nn_graph();
  }

  std::shared_ptr<LazyJobInstance> MakeJobInstance(Instruction* instruction) const {
    const auto* ptr = instruction->phy_instr_operand().get();
    const auto* phy_instr_operand = dynamic_cast<const LaunchLazyJobPhyInstrOperand*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    const auto& nn_graph = phy_instr_operand->nn_graph();
    const auto& FinishCb = [this, instruction]() {
      auto* device_ctx = GetLazyJobDeviceCtx(instruction);
      device_ctx->DequeueNNGraph();
      auto* status_buffer = instruction->mut_status_buffer();
      NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer())->set_done();
    };
    return std::make_shared<LazyJobInstance>(nn_graph->job_name(), FinishCb);
  }
};

}  // namespace vm
}  // namespace oneflow
#endif  // ONEFLOW_CORE_EAGER_LAZY_JOB_INSTRUCTION_TYPE_H_
