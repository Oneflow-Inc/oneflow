#include "oneflow/core/actor/reduce_ag_actor.h"

namespace oneflow {

void ReduceAGActor::VirtualActorInit(const TaskProto& task_proto) {
  consumed_regst_num_ = task_proto.consumed_regst_desc_id().size();
  processed_regst_cnt_ = 0;
  regsts_in_using_.clear();
  regsts_used_.clear();
  OF_SET_MSG_HANDLER(&ReduceAGActor::HandlerNormal);
}

bool ReduceAGActor::IsReadReady() {
  std::vector<int64_t> regst_desc_id_vec = GetNaiveReadableRegstDescIdVec();
  CHECK(regsts_in_using_.empty());
  int64_t cur_piece_id = processed_regst_cnt_ / consumed_regst_num_;
  HashSet<Regst*>& cur_piece_regsts_used = regsts_used_[cur_piece_id];
  for (int64_t regst_desc_id : regst_desc_id_vec) {
    Regst* regst = GetNaiveCurReadable(regst_desc_id);
    if (regst->piece_id() != cur_piece_id
        || cur_piece_regsts_used.find(regst) != cur_piece_regsts_used.end()) {
      continue;
    }
    CHECK(regsts_in_using_.emplace(regst->regst_desc_id(), regst).second);
    CHECK(cur_piece_regsts_used.emplace(regst).second);
  }
  return !regsts_in_using_.empty();
}

void ReduceAGActor::Act(std::function<bool(Regst*)>* IsNaiveAllowedReturnToProducer) {
  int64_t cur_piece_id = processed_regst_cnt_ / consumed_regst_num_;
  int64_t actor_id = this->actor_id();
  HashMap<int64_t, Regst*> regsts_in_using(regsts_in_using_);
  *IsNaiveAllowedReturnToProducer = [actor_id, cur_piece_id, regsts_in_using](Regst* regst) {
    if (regsts_in_using.find(regst->regst_desc_id()) != regsts_in_using.end()) {
      return true;
    } else {
      return false;
    }
  };
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = reinterpret_cast<void*>(processed_regst_cnt_);
  for (const ExecKernel& ek : exec_kernel_vec()) {
    ek.kernel->Launch(kernel_ctx, [&](const std::string& bn_in_op) -> Blob* {
      auto regst_desc_id_it = ek.bn_in_op2regst_desc_id.find(bn_in_op);
      if (regst_desc_id_it == ek.bn_in_op2regst_desc_id.end()) { return nullptr; }
      Regst* regst = GetCurWriteableRegst(regst_desc_id_it->second);
      if (regsts_in_using_.find(regst_desc_id_it->second) != regsts_in_using_.end()) {
        regst = regsts_in_using_.at(regst_desc_id_it->second);
      }
      if (regst == nullptr) { return nullptr; }
      const LogicalBlobId& lbi = ek.kernel->BnInOp2Lbi(bn_in_op);
      return regst->GetBlobByLbi(lbi);
    });
  }
  processed_regst_cnt_ += regsts_in_using_.size();
  if (processed_regst_cnt_ % consumed_regst_num_ == 0) {
    AsyncSendRegstMsgToConsumer([&](Regst* regst) {
      regst->set_piece_id(cur_piece_id);
      return true;
    });
    CHECK_EQ(regsts_used_.at(cur_piece_id).size(), consumed_regst_num_);
    regsts_used_.erase(cur_piece_id);
  }
  regsts_in_using_.clear();
}

REGISTER_ACTOR(TaskType::kReduceGather, ReduceAGActor);
REGISTER_ACTOR(TaskType::kReduceAdd, ReduceAGActor);

}  // namespace oneflow
