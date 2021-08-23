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

#include "oneflow/core/eager/lazy_job_stream_type.h"
#include "oneflow/core/eager/lazy_job_device_context.h"
#include "oneflow/core/eager/run_lazy_job_phy_instr_operand.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"

namespace oneflow {

namespace {

class LazyJobInstance final : public JobInstance {
 public:
  LazyJobInstance(const LazyJobInstance&) = delete;
  LazyJobInstance(LazyJobInstance&&) = delete;
  ~LazyJobInstance() override = default;
  LazyJobInstance(const std::string& job_name,
                  const HashMap<std::string, std::function<void(int64_t)>>& push_cbs,
                  const HashMap<std::string, std::function<void(int64_t)>>& pull_cbs,
                  const std::function<void()> finish_cb)
      : job_name_(job_name), push_cbs_(push_cbs), pull_cbs_(pull_cbs), finish_cb_(finish_cb) {}

  std::string job_name() const override { return job_name_; }
  void PushBlobByOpName(uint64_t ofblob_ptr, const std::string& op_name) const override {
    const auto& push_cb = CHECK_JUST(MapAt(push_cbs_, op_name));
    return push_cb(ofblob_ptr);
  }
  void PullBlobByOpName(uint64_t ofblob_ptr, const std::string& op_name) const override {
    const auto& pull_cb = CHECK_JUST(MapAt(pull_cbs_, op_name));
    return pull_cb(ofblob_ptr);
  }
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
  const HashMap<std::string, std::function<void(int64_t)>> push_cbs_;
  const HashMap<std::string, std::function<void(int64_t)>> pull_cbs_;
  const std::function<void()> finish_cb_;
};

}  // namespace

namespace vm {

class RunLazyJobInstructionType final : public InstructionType {
 public:
  RunLazyJobInstructionType(const RunLazyJobInstructionType&) = delete;
  RunLazyJobInstructionType(RunLazyJobInstructionType&&) = delete;
  RunLazyJobInstructionType() = default;
  ~RunLazyJobInstructionType() = default;
  using stream_type = LazyJobStreamType;
  void Infer(vm::Instruction* instruction) const override { UNIMPLEMENTED(); }
  void Compute(vm::Instruction* instruction) const override {
    const auto& cur_nn_graph = GetCurNNGraph(instruction);
    auto* device_ctx = GetLazyJobDeviceCtx(instruction);

    device_ctx->WaitUntilQueueEmptyIfFrontNNGraphNotEquals(cur_nn_graph);
    {
      const auto& job_instance = MakeJobInstance(instruction);
      const auto& job_name = job_instance->job_name();
      auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
      for (const auto& op_name : cur_nn_graph->inputs_op_names()) {
        buffer_mgr->Get(GetInputBufferName(job_name, op_name))->Send(job_instance);
      }
      for (const auto& op_name : cur_nn_graph->outputs_op_names()) {
        buffer_mgr->Get(GetOutputBufferName(job_name, op_name))->Send(job_instance);
      }
      buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Send(job_instance);
      buffer_mgr->Get(GetSourceTickBufferName(job_name))->Send(job_instance);
    }
    device_ctx->EnqueueNNGraph(cur_nn_graph);
  }

 private:
  LazyJobDeviceCtx* GetLazyJobDeviceCtx(Instruction* instruction) const {
    auto* stream = instruction->mut_stream();
    auto* device_ctx = dynamic_cast<LazyJobDeviceCtx*>(stream->device_ctx().get());
    CHECK_NOTNULL(device_ctx);
    return device_ctx;
  }

  std::shared_ptr<NNGraphIf> GetCurNNGraph(Instruction* instruction) const {
    const auto* ptr = instruction->instr_msg().phy_instr_operand().get();
    const auto* phy_instr_operand = dynamic_cast<const RunLazyJobPhyInstrOperand*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    return phy_instr_operand->nn_graph();
  }

  std::shared_ptr<LazyJobInstance> MakeJobInstance(Instruction* instruction) const {
    const auto* ptr = instruction->instr_msg().phy_instr_operand().get();
    const auto* phy_instr_operand = dynamic_cast<const RunLazyJobPhyInstrOperand*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    const auto& nn_graph = phy_instr_operand->nn_graph();
    HashMap<std::string, std::function<void(int64_t)>> push_cbs;
    CHECK_EQ(nn_graph->inputs_op_names().size(), phy_instr_operand->inputs()->size());
    for (int i = 0; i < nn_graph->inputs_op_names().size(); ++i) {
      const auto* blob = &phy_instr_operand->inputs()->at(i)->blob();
      if (!blob) { continue; }
      const auto& op_name = nn_graph->inputs_op_names().at(i);
      const auto& PushCb = [blob](int64_t of_blob_ptr) {
        OfBlob* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
        of_blob->mut_blob()->CopyHeaderFrom(of_blob->mut_device_ctx(), blob);
        of_blob->mut_blob()->CopyDataContentFrom(of_blob->mut_device_ctx(), blob);
      };
      CHECK(push_cbs.emplace(op_name, PushCb).second);
    }
    HashMap<std::string, std::function<void(int64_t)>> pull_cbs;
    CHECK_EQ(nn_graph->outputs_op_names().size(), phy_instr_operand->outputs()->size());
    for (int i = 0; i < nn_graph->outputs_op_names().size(); ++i) {
      auto* mut_blob = phy_instr_operand->outputs()->at(i)->mut_blob();
      if (!mut_blob) { continue; }
      const auto& op_name = nn_graph->outputs_op_names().at(i);
      const auto& PullCb = [mut_blob](int64_t of_blob_ptr) {
        OfBlob* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
        mut_blob->CopyHeaderFrom(of_blob->mut_device_ctx(), &of_blob->blob());
        mut_blob->CopyDataContentFrom(of_blob->mut_device_ctx(), &of_blob->blob());
      };
      CHECK(pull_cbs.emplace(op_name, PullCb).second);
    }
    const auto& FinishCb = [this, instruction]() {
      auto* device_ctx = GetLazyJobDeviceCtx(instruction);
      device_ctx->DequeueNNGraph();
      auto* status_buffer = instruction->mut_status_buffer();
      NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
    };
    return std::make_shared<LazyJobInstance>(nn_graph->job_name(), push_cbs, pull_cbs, FinishCb);
  }
};

COMMAND(RegisterInstructionType<RunLazyJobInstructionType>("RunLazyJob"));

}  // namespace vm
}  // namespace oneflow
