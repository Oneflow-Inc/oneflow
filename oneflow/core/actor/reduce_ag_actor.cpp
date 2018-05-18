#include "oneflow/core/actor/reduce_ag_actor.h"

namespace oneflow {

void ReduceAGActor::VirtualActorInit(const TaskProto& task_proto) {
  for (auto& pair : task_proto.consumed_regst_desc_id()) {
    CHECK_EQ(pair.second.regst_desc_id().size(), 1);
    for (int64_t regst_desc_id : pair.second.regst_desc_id()) { in_regsts_[regst_desc_id] = {}; }
  }
  processed_regst_cnt_ = 0;
  regsts_in_using_.clear();
  OF_SET_MSG_HANDLER(&ReduceAGActor::HandlerNormal);
}

bool ReduceAGActor::IsCustomizedReadReady() {
  CHECK(regsts_in_using_.empty());
  int64_t cur_piece_id = processed_regst_cnt_ / in_regsts_.size();
  for (auto& pair : in_regsts_) {
    if (pair.second.empty()) { continue; }
    Regst* regst = pair.second.front();
    CHECK_GE(regst->piece_id(), cur_piece_id);
    if (regst->piece_id() > cur_piece_id) { continue; }
    CHECK(regsts_in_using_.emplace(regst->regst_desc_id(), regst).second);
    pair.second.pop();
  }
  return !regsts_in_using_.empty();
}

void ReduceAGActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  Regst* regst = msg.regst();
  auto in_regsts_it = in_regsts_.find(regst->regst_desc_id());
  CHECK(in_regsts_it != in_regsts_.end());
  in_regsts_it->second.push(regst);
  LOG(INFO) << actor_id() << " receive " << regst->regst_desc_id() << " at " << regst->piece_id();
}

void ReduceAGActor::Act(std::function<bool(Regst*)>* IsNaiveAllowedReturnToProducer) {
  int64_t cur_piece_id = processed_regst_cnt_ / in_regsts_.size();
  *IsNaiveAllowedReturnToProducer = [](Regst* regst) { return false; };
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
  for (auto pair : regsts_in_using_) { AsyncSendRegstMsgToProducer(pair.second); }
  regsts_in_using_.clear();
  if (processed_regst_cnt_ % in_regsts_.size() == 0) {
    AsyncSendRegstMsgToConsumer([&](Regst* regst) {
      LOG(INFO) << actor_id() << " send " << regst->regst_desc_id() << " at " << cur_piece_id;
      regst->set_piece_id(cur_piece_id);
      return true;
    });
  }
}

REGISTER_ACTOR(TaskType::kReduceGather, ReduceAGActor);
REGISTER_ACTOR(TaskType::kReduceAdd, ReduceAGActor);

}  // namespace oneflow
