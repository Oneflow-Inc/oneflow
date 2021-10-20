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
#include "oneflow/core/eager/lazy_job_phy_instr_operand.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/kernel/kernel_util.h"

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

class LaunchLazyJobInstructionType final : public InstructionType {  // NOLINT
 public:
  LaunchLazyJobInstructionType(const LaunchLazyJobInstructionType&) = delete;
  LaunchLazyJobInstructionType(LaunchLazyJobInstructionType&&) = delete;
  LaunchLazyJobInstructionType() = default;
  ~LaunchLazyJobInstructionType() = default;
  using stream_type = LazyJobStreamType;
  void Infer(vm::Instruction* instruction) const override { UNIMPLEMENTED(); }
  void Compute(vm::Instruction* instruction) const override {
    const auto* ptr = instruction->instr_msg().phy_instr_operand().get();
    const auto* phy_instr_operand = dynamic_cast<const LaunchLazyJobPhyInstrOperand*>(ptr);
    const auto& cur_nn_graph = GetCurNNGraph(instruction);
    auto* device_ctx = GetLazyJobDeviceCtx(instruction);

    OF_PROFILER_RANGE_PUSH("WaitUntilQueueEmptyIfFrontNNGraphNotEquals");
    device_ctx->WaitUntilQueueEmptyIfFrontNNGraphNotEquals(cur_nn_graph);
    OF_PROFILER_RANGE_POP();  // WaitUntilQueueEmptyIfFrontNNGraphNotEquals
    {
      OF_PROFILER_RANGE_PUSH("MakeJobInstance");
      const auto& job_instance = MakeJobInstance(instruction);
      OF_PROFILER_RANGE_POP();  // MakeJobInstance
      OF_PROFILER_RANGE_PUSH("Send all buffers to BufferMgr");
      const auto& job_name = job_instance->job_name();
      auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
      for (int i = 0; i < cur_nn_graph->inputs_op_names().size(); ++i) {
        if (cur_nn_graph->inputs_valid().at(i)) {
          const std::string& input_op_name = cur_nn_graph->inputs_op_names().at(i);
          buffer_mgr->Get(GetInputBufferName(job_name, input_op_name))->Push(job_instance);
        }
      }
      for (int i = 0; i < cur_nn_graph->outputs_op_names().size(); ++i) {
        if (cur_nn_graph->outputs_valid().at(i)) {
          const std::string& output_op_name = cur_nn_graph->outputs_op_names().at(i);
          buffer_mgr->Get(GetOutputBufferName(job_name, output_op_name))->Push(job_instance);
        }
      }
      buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Push(job_instance);
      buffer_mgr->Get(GetSourceTickBufferName(job_name))->Push(job_instance);
      OF_PROFILER_RANGE_POP();  // BufferMgr
    }
    OF_PROFILER_RANGE_PUSH("EnqueueNNGraph");
    device_ctx->EnqueueNNGraph(cur_nn_graph);
    OF_PROFILER_RANGE_POP();  // EnqueueNNGraph
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
    const auto* phy_instr_operand = dynamic_cast<const LaunchLazyJobPhyInstrOperand*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    return phy_instr_operand->nn_graph();
  }

  std::shared_ptr<LazyJobInstance> MakeJobInstance(Instruction* instruction) const {
    const auto* ptr = instruction->instr_msg().phy_instr_operand().get();
    const auto* phy_instr_operand = dynamic_cast<const LaunchLazyJobPhyInstrOperand*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    const auto& nn_graph = phy_instr_operand->nn_graph();
    HashMap<std::string, std::function<void(int64_t)>> push_cbs;
    for (int i = 0; i < nn_graph->inputs_op_names().size(); ++i) {
      const auto& input_op_name = nn_graph->inputs_op_names().at(i);
      const auto& end_event_record =
          CHECK_JUST(phy_instr_operand->EndEventRecord4OpName(input_op_name));
      if (nn_graph->inputs_valid().at(i)) {
        const auto& input_blob_object = phy_instr_operand->input_blob_objects()->at(i);
        const auto& PushCb = [input_op_name, end_event_record,
                              input_blob_object](int64_t of_blob_ptr) {
          OfBlob* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
          const Blob* blob = &input_blob_object->blob();
          CHECK_NOTNULL(blob);
          of_blob->mut_blob()->CopyHeaderFrom(of_blob->mut_device_ctx(), blob);
          if (blob->dptr() == nullptr) {
            end_event_record->Init(std::make_shared<NaiveEventRecord>());
          } else {
            AutoMemcpy(of_blob->mut_device_ctx(), of_blob->mut_blob(), blob);
            auto* event_record_provider =
                CHECK_NOTNULL(dynamic_cast<EventRecordProvider*>(of_blob->mut_device_ctx()));
            end_event_record->Init(event_record_provider->MakeEventRecord());
          }
        };
        CHECK(push_cbs.emplace(input_op_name, PushCb).second);
      } else {
        end_event_record->Init(std::make_shared<NaiveEventRecord>());
      }
    }
    HashMap<std::string, std::function<void(int64_t)>> pull_cbs;
    for (int i = 0; i < nn_graph->outputs_op_names().size(); ++i) {
      const auto& output_op_name = nn_graph->outputs_op_names().at(i);
      const auto& end_event_record =
          CHECK_JUST(phy_instr_operand->EndEventRecord4OpName(output_op_name));
      if (nn_graph->outputs_valid().at(i)) {
        const auto& output_blob_object = phy_instr_operand->output_blob_objects()->at(i);
        const auto& PullCb = [output_op_name, end_event_record,
                              output_blob_object](int64_t of_blob_ptr) {
          OfBlob* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
          Blob* mut_blob = output_blob_object->mut_blob();
          CHECK_NOTNULL(mut_blob);
          mut_blob->CopyHeaderFrom(of_blob->mut_device_ctx(), &of_blob->blob());
          if (mut_blob->dptr() == nullptr) {
            end_event_record->Init(std::make_shared<NaiveEventRecord>());
          } else {
            AutoMemcpy(of_blob->mut_device_ctx(), mut_blob, &of_blob->blob());
            auto* event_record_provider =
                CHECK_NOTNULL(dynamic_cast<EventRecordProvider*>(of_blob->mut_device_ctx()));
            end_event_record->Init(event_record_provider->MakeEventRecord());
          }
        };
        CHECK(pull_cbs.emplace(output_op_name, PullCb).second);
      } else {
        end_event_record->Init(std::make_shared<NaiveEventRecord>());
      }
    }
    const auto& op_name2end_event_record = phy_instr_operand->op_name2end_event_record();
    const auto& FinishCb = [this, instruction, op_name2end_event_record]() {
      for (const auto& pair : *op_name2end_event_record) {
        pair.second->TryInit(std::make_shared<NaiveEventRecord>());
      }
      auto* device_ctx = GetLazyJobDeviceCtx(instruction);
      device_ctx->DequeueNNGraph();
      auto* status_buffer = instruction->mut_status_buffer();
      NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
    };
    return std::make_shared<LazyJobInstance>(nn_graph->job_name(), push_cbs, pull_cbs, FinishCb);
  }
};

COMMAND(RegisterInstructionType<LaunchLazyJobInstructionType>("LaunchLazyJob"));

}  // namespace vm
}  // namespace oneflow
