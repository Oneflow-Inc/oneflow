#include "oneflow/core/actor/reduce_actor.h"

namespace oneflow {

void ReduceActor::VirtualActorInit(const TaskProto& task_proto) {
  consumed_regst_num_ = task_proto.consumed_regst_desc_id().size();
  processed_regst_cnt_ = 0;
  OF_SET_MSG_HANDLER(&ReduceActor::HandlerNormal);
}

bool ReduceActor::IsReadReady() { return !GetNaiveReadableRegstDescIdVec().empty(); }

void ReduceActor::Act(std::function<bool(Regst*)>* IsNaiveAllowedReturnToProducer) {
  std::vector<int64_t> regst_desc_id_vec = GetNaiveReadableRegstDescIdVec();
  HashSet<Regst*> cur_used_regsts;
  int64_t cur_piece_id = processed_regst_cnt_ / consumed_regst_num_;
  for (int64_t regst_desc_id : regst_desc_id_vec) {
    Regst* regst = GetNaiveCurReadable(regst_desc_id);
    CHECK_GE(regst->piece_id(), cur_piece_id);
    if (regst->piece_id() > cur_piece_id) { continue; }
    CHECK(cur_used_regsts.emplace(regst).second);
  }
  *IsNaiveAllowedReturnToProducer = [cur_used_regsts](Regst* regst) {
    if (cur_used_regsts.find(regst) != cur_used_regsts.end()) {
      return true;
    } else {
      return false;
    }
  };
  if (cur_used_regsts.empty()) { return; }
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = reinterpret_cast<void*>(processed_regst_cnt_);
  AsyncLaunchKernel(kernel_ctx,
                    [cur_used_regsts](int64_t regst_desc_id) { return *cur_used_regsts.begin(); });
  processed_regst_cnt_ += cur_used_regsts.size();
  if (processed_regst_cnt_ % consumed_regst_num_ == 0) {
    AsyncSendRegstMsgToConsumer([&](Regst* regst) {
      regst->set_piece_id(cur_piece_id + 1);
      return true;
    });
  }
}

REGISTER_ACTOR(TaskType::kReduceAdd, ReduceActor);
REGISTER_ACTOR(TaskType::kReduceGather, ReduceActor);

}  // namespace oneflow
