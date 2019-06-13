#include "oneflow/core/actor/case_compute_actor.h"

namespace oneflow {

void CaseCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  CHECK_EQ(1, exec_kernel_vec().size());
  cur_selected_id_ = -1;
  const int32_t output_bns_size =
      task_proto.exec_sequence().exec_node().Get(0).kernel_conf().op_attribute().output_bns_size();
  FOR_RANGE(int64_t, i, 0, output_bns_size) {
    const int64_t regst_desc_id =
        exec_kernel_vec().at(0).bn_in_op2regst_desc_id.at(GenRepeatedBn("out", i));
    CHECK(out_bn_id2regst_desc_id_.emplace(i, regst_desc_id).second);
  }
  OF_SET_MSG_HANDLER(&CaseCompActor::HandlerNormal);
}

void CaseCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &cur_selected_id_;
  AsyncLaunchKernel(kernel_ctx);
}

void CaseCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
    const int64_t selected_regst_desc_id = out_bn_id2regst_desc_id_.at(cur_selected_id_);
    if (regst->regst_desc_id() == selected_regst_desc_id) {
      regst->set_piece_id(++regst_desc_id2piece_id_[selected_regst_desc_id]);
      return true;
    } else {
      return false;
    }
  });
  cur_selected_id_ = -1;
}

bool CaseCompActor::ProducedCtrlRegstValid(int64_t regst_desc_id) const { return true; }

bool CaseCompActor::CheckOutputActId(int64_t regst_desc_id) const { return false; }

REGISTER_ACTOR(kCase, CaseCompActor);

}  // namespace oneflow
