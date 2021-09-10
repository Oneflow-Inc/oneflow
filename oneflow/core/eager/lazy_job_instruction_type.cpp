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
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {

namespace {

bool IsEmptyComponentEagerBlobObject(
    const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) {
  Blob* blob = eager_blob_object->mut_blob();
  if (blob && blob->shape().elem_cnt() > 0 && blob->dptr()) {
    // NOTE(chengcheng):
    //   Scalar shape has elem_cnt == 1;
    //   0-Size shape has NumAxes > 0 and has dim value 0.
    return false;
  }
  return true;
}

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

  bool HasPushCbOpName(const std::string& op_name) const {
    int64_t this_rank = GlobalProcessCtx::Rank();
    LOG(ERROR) << "rank = " << this_rank << " push_cbs_.size = " << push_cbs_.size() << "\n";
    for (auto& pair : push_cbs_) {
      LOG(ERROR) << "rank = " << this_rank << " : input_op_name = " << pair.first << "\n";
    }
    return push_cbs_.find(op_name) != push_cbs_.end();
  }

  bool HasPullCbOpName(const std::string& op_name) const {
    int64_t this_rank = GlobalProcessCtx::Rank();
    LOG(ERROR) << "rank = " << this_rank << " pull_cbs_.size = " << pull_cbs_.size() << "\n";
    for (auto& pair : pull_cbs_) {
      LOG(ERROR) << "rank = " << this_rank << " : output_op_name = " << pair.first << "\n";
    }
    return pull_cbs_.find(op_name) != pull_cbs_.end();
  }

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

    OF_PROFILER_RANGE_PUSH("WaitUntilQueueEmptyIfFrontNNGraphNotEquals");
    device_ctx->WaitUntilQueueEmptyIfFrontNNGraphNotEquals(cur_nn_graph);
    OF_PROFILER_RANGE_POP();  // WaitUntilQueueEmptyIfFrontNNGraphNotEquals
    {
      OF_PROFILER_RANGE_PUSH("MakeJobInstance");
      const auto& job_instance = MakeJobInstance(instruction);
      OF_PROFILER_RANGE_POP();  // MakeJobInstance
      OF_PROFILER_RANGE_PUSH("Send all buffers to BufferMgr");
      const auto& job_name = job_instance->job_name();
      int64_t this_rank = GlobalProcessCtx::Rank();
      auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
      for (const auto& op_name : cur_nn_graph->inputs_op_names()) {
        if (job_instance->HasPushCbOpName(op_name)) {
          LOG(ERROR) << "cclog: in rank: " << this_rank << " has input with op name: " << op_name
                     << " so send push cb.\n";
          buffer_mgr->Get(GetInputBufferName(job_name, op_name))->Send(job_instance);
        } else {
          LOG(ERROR) << "cclog: in rank: " << this_rank
                     << " has NOT input with op name: " << op_name << " so NOT send push cb.\n";
        }
      }
      for (const auto& op_name : cur_nn_graph->outputs_op_names()) {
        if (job_instance->HasPullCbOpName(op_name)) {
          LOG(ERROR) << "cclog: in rank: " << this_rank << " has output with op name: " << op_name
                     << " so send pull cb.\n";
          buffer_mgr->Get(GetOutputBufferName(job_name, op_name))->Send(job_instance);
        } else {
          LOG(ERROR) << "cclog: in rank: " << this_rank
                     << " has NOT output with op name: " << op_name << " so NOT send pull cb.\n";
        }
      }
      buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Send(job_instance);
      buffer_mgr->Get(GetSourceTickBufferName(job_name))->Send(job_instance);
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
    const auto* phy_instr_operand = dynamic_cast<const RunLazyJobPhyInstrOperand*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    return phy_instr_operand->nn_graph();
  }

  std::shared_ptr<LazyJobInstance> MakeJobInstance(Instruction* instruction) const {
    int64_t this_rank = GlobalProcessCtx::Rank();
    const auto* ptr = instruction->instr_msg().phy_instr_operand().get();
    const auto* phy_instr_operand = dynamic_cast<const RunLazyJobPhyInstrOperand*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    const auto& nn_graph = phy_instr_operand->nn_graph();
    HashMap<std::string, std::function<void(int64_t)>> push_cbs;
    CHECK_EQ(nn_graph->inputs_op_names().size(), phy_instr_operand->inputs()->size());
    for (int i = 0; i < nn_graph->inputs_op_names().size(); ++i) {
      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object =
          phy_instr_operand->inputs()->at(i);
      const auto* blob = &eager_blob_object->blob();
      const auto& op_name = nn_graph->inputs_op_names().at(i);
      if (IsEmptyComponentEagerBlobObject(eager_blob_object)) {
        LOG(ERROR) << " in rank = " << this_rank << ", op_name: " << op_name
                   << " SKIP build push cb.";
        continue;
      }
      const auto& PushCb = [blob](int64_t of_blob_ptr) {
        OfBlob* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
        std::cout << "cclog: in PushCb, input tensor blob: header_size = "
                  << blob->blob_desc().ByteSizeOfBlobHeader()
                  << " header_shape = " << blob->shape().ToString()
                  << " aligned header size = " << blob->blob_desc().AlignedByteSizeOfBlobHeader()
                  << " \n input regst blob: header_size = "
                  << of_blob->mut_blob()->blob_desc().ByteSizeOfBlobHeader()
                  << " header_shape = " << of_blob->mut_blob()->shape().ToString()
                  << ": aligned header size = "
                  << of_blob->mut_blob()->blob_desc().AlignedByteSizeOfBlobHeader() << "\n";
        of_blob->mut_blob()->CopyHeaderFrom(of_blob->mut_device_ctx(), blob);
        of_blob->mut_blob()->CopyDataContentFrom(of_blob->mut_device_ctx(), blob);
      };
      CHECK(push_cbs.emplace(op_name, PushCb).second);
      LOG(ERROR) << " in rank = " << this_rank << ", op_name: " << op_name << " build push cb.";
    }
    HashMap<std::string, std::function<void(int64_t)>> pull_cbs;
    CHECK_EQ(nn_graph->outputs_op_names().size(), phy_instr_operand->outputs()->size());
    for (int i = 0; i < nn_graph->outputs_op_names().size(); ++i) {
      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object =
          phy_instr_operand->outputs()->at(i);
      const auto& op_name = nn_graph->outputs_op_names().at(i);
      auto* mut_blob = eager_blob_object->mut_blob();
      if (IsEmptyComponentEagerBlobObject(eager_blob_object)) {
        LOG(ERROR) << " in rank = " << this_rank << ", op_name: " << op_name
                   << " SKIP build pull cb.";
        continue;
      }
      const auto& PullCb = [mut_blob](int64_t of_blob_ptr) {
        OfBlob* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
        mut_blob->CopyHeaderFrom(of_blob->mut_device_ctx(), &of_blob->blob());
        mut_blob->CopyDataContentFrom(of_blob->mut_device_ctx(), &of_blob->blob());
      };

      LOG(ERROR) << " in rank = " << this_rank << ", op_name: " << op_name << " build pull cb.";
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
