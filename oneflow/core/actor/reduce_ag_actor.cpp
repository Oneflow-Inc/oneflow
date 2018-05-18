#include "oneflow/core/actor/reduce_ag_actor.h"

namespace oneflow {

void ReduceAGActor::VirtualActorInit(const TaskProto& task_proto) {
  in_regsts_.clear();
  for (auto& pair : task_proto.consumed_regst_desc_id()) {
    CHECK_EQ(pair.second.regst_desc_id().size(), 1);
    in_regsts_[pair.second.regst_desc_id().Get(0)] = {};
  }
  processed_regsts_cnt_ = 0;
  in_regsts_eord_cnt_ = 0;
  ready_in_regsts_.clear();
  OF_SET_MSG_HANDLER(&ReduceAGActor::HandlerNormal);
}

bool ReduceAGActor::IsCustomizedReadReady() {
  int64_t cur_piece_id = processed_regsts_cnt_ / in_regsts_.size();
  ready_in_regsts_.clear();
  for (auto& pair : in_regsts_) {
    if (pair.second.empty()) { continue; }
    Regst* regst = pair.second.front();
    CHECK_EQ(pair.first, regst->regst_desc_id());
    CHECK_GE(regst->piece_id(), cur_piece_id);
    if (regst->piece_id() == cur_piece_id) {
      CHECK(ready_in_regsts_.emplace(regst->regst_desc_id(), regst).second);
    }
  }
  return !ready_in_regsts_.empty();
}

bool ReduceAGActor::IsCustomizedReadAlwaysUnReadyFromNow() {
  return in_regsts_eord_cnt_ == in_regsts_.size();
}

void ReduceAGActor::NormalProcessCustomizedEordMsg(const ActorMsg& msg) {
  CHECK(in_regsts_.find(msg.eord_regst_desc_id()) != in_regsts_.end());
  ++in_regsts_eord_cnt_;
}

void ReduceAGActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  Regst* regst = msg.regst();
  auto in_regsts_it = in_regsts_.find(regst->regst_desc_id());
  CHECK(in_regsts_it != in_regsts_.end());
  in_regsts_it->second.push(regst);
}

void ReduceAGActor::Act() {
  int64_t cur_piece_id = processed_regsts_cnt_ / in_regsts_.size();
  for (auto& pair : ready_in_regsts_) {
    CHECK(!in_regsts_.at(pair.first).empty());
    in_regsts_.at(pair.first).pop();
  }
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = reinterpret_cast<void*>(processed_regsts_cnt_);
  HashMap<int64_t, Regst*> ready_in_regsts(ready_in_regsts_);
  for (const ExecKernel& ek : exec_kernel_vec()) {
    ek.kernel->Launch(kernel_ctx, [&, ready_in_regsts](const std::string& bn_in_op) -> Blob* {
      auto regst_desc_id_it = ek.bn_in_op2regst_desc_id.find(bn_in_op);
      if (regst_desc_id_it == ek.bn_in_op2regst_desc_id.end()) { return nullptr; }
      Regst* regst = GetCurWriteableRegst(regst_desc_id_it->second);
      if (ready_in_regsts.find(regst_desc_id_it->second) != ready_in_regsts.end()) {
        regst = ready_in_regsts.at(regst_desc_id_it->second);
      }
      if (regst == nullptr) { return nullptr; }
      const LogicalBlobId& lbi = ek.kernel->BnInOp2Lbi(bn_in_op);
      return regst->GetBlobByLbi(lbi);
    });
  }
  processed_regsts_cnt_ += ready_in_regsts_.size();
  if (processed_regsts_cnt_ % in_regsts_.size() == 0) {
    AsyncSendRegstMsgToConsumer([&](Regst* regst) {
      regst->set_piece_id(cur_piece_id);
      return true;
    });
  }
  for (auto pair : ready_in_regsts_) { AsyncSendRegstMsgToProducer(pair.second); }
  ready_in_regsts_.clear();
}

REGISTER_ACTOR(TaskType::kReduceGather, ReduceAGActor);
REGISTER_ACTOR(TaskType::kReduceAdd, ReduceAGActor);

}  // namespace oneflow
