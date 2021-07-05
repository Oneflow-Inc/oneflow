#include "oneflow/eager/lazy_job_stream_type.h"
#include "oneflow/framework/nn_graph_if.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"

namespace oneflow {
namespace vm {

class RunJobInstructionType : public InstructionType {
 public:
  using stream_type = LazyJobStreamType;
  void Infer(vm::Instruction* instruction) const override {
    UNIMPLEMENTED();
  }
  void Compute(vm::Instruction* instruction) const override {
    const auto& cur_nn_graph = GetCurNNGraph(instruction);
    auto* device_ctx = GetLazyJobDeviceCtx(instruction);

    device_ctx->WaitUntilQeueEmptyIfFrontNNGraphNotEquals(cur_nn_graph);
    {
      const auto& job_instance = MakeJobInstance(instruction);
      const auto& job_name = job_instance->job_name();
      auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Get();
      if (job_instance->has_inputs()) {
        buffer_mgr->Get(GetForeignOutputBufferName(job_name))->Send(job_instance);
      }
      if (job_instance->has_outputs()) {
        buffer_mgr->Get(GetForeignInputBufferName(job_name))->Send(job_instance);
      }
      buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Send(job_instance);
      buffer_mgr->Get(GetSourceTickBufferName(job_name))->Send(job_instance);
    }
    device_ctx->EnqueueNNGraph(cur_nn_graph);
  }

 private:
  virtual const char* device_tag() const { return "lazy_job"; }

  LazyJobDeviceCtx* GetLazyJobDeviceCtx(Instruction* instruction) const {
    auto* stream = instruction->mut_stream();
    auto* device_ctx = dynamic_cast<LazyJobDeviceCtx*>(stream->device_ctx().get());
    CHECK_NOTNULL(device_ctx);
    return device_ctx;
  }

  std::shared_ptr<NNGraphIf> GetCurNNGraph(Instruction* instruction) const {
    const auto* phy_instr_operand = dynamic_cast<const RunJobPhyInstrOperand*>(instruction.phy_instr_operand().get());
    CHECK_NOTNULL(phy_instr_operand);
    return phy_instr_operand->nn_graph();
  }

  static std::shared_ptr<LazyJobInstance> MakeJobInstance(Instruction* instruction) const {
    const auto* phy_instr_operand = dynamic_cast<const RunJobPhyInstrOperand*>(instruction.phy_instr_operand().get());
    const auto& nn_graph = phy_instr_operand->nn_graph();
    HashMap<std::string, std::function<void(int64_t)>> push_cbs;
    CHECK_EQ(nn_graph.inputs_op_names().size() == phy_instr_operand.inputs().size());
    for (int i = 0; i < nn_graph.inputs_op_names().size(); ++i) {
      const auto& op_name = nn_graph.inputs_op_names().at(i);
      const auto* blob = &phy_instr_operand.inputs().at(i).blob();
      const auto& PushCb = [blob](int64_t of_blob_ptr) {
        OfBlob* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
        of_blob->mut_blob()->CopyHeaderFrom(of_blob->mut_device_ctx(), blob);
        of_blob->mut_blob()->CopyDataContentFrom(of_blob->mut_device_ctx(), blob);
      };
      CHECK(push_cbs.emplace(op_name, PushCb).second);
    }
    HashMap<std::string, std::function<void(int64_t)>> pull_cbs;
    CHECK_EQ(nn_graph.outputs_op_names().size() == phy_instr_operand.outputs().size());
    for (int i = 0; i < nn_graph.outputs_op_names().size(); ++i) {
      const auto& op_name = nn_graph.outputs_op_names().at(i);
      auto* mut_blob = phy_instr_operand.outputs().at(i)->mut_blob();
      const auto& PullCb = [mut_blob](int64_t of_blob_ptr) {
        OfBlob* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
        mut_blob->CopyHeaderFrom(of_blob->mut_device_ctx(), of_blob->blob());
        mut_blob->CopyDataContentFrom(of_blob->mut_device_ctx(), of_blob->blob());
      };
      CHECK(pull_cbs.emplace(op_name, PullCb).second);
    }
    const auto& FinishCb = [instruction]() {
      auto* device_ctx = GetLazyJobDeviceCtx(instruction);
      device_ctx->DequeueNNGraph();
      auto* status_buffer = instruction->mut_status_buffer();
      NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
    };
    return std::make_shared<LazyJobInstance>(nn_graph->job_name(), push_cbs, pull_cbs, FinishCb);
  }
};
 
}
}
