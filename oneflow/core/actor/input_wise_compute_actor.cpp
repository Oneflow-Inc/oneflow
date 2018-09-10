#include "oneflow/core/actor/input_wise_compute_actor.h"

namespace oneflow {

void InputWiseCompActor::Init(const TaskProto& task_proto) {
  CHECK_EQ(1, exec_kernel_vec().size());
  const auto& input_bns =
      task_proto.exec_sequence().exec_node().Get(0).kernel_conf().op_attribute().input_bns();
  HashMap<std::string, int64_t> ibn2in_bn_id;
  for (int64_t i = 0; i < input_bns.size(); ++i) {
    CHECK(ibn2in_bn_id.emplace(input_bns.Get(i), i).second);
  }
  for (const auto& pair : exec_kernel_vec().at(0).bn_in_op2regst_desc_id) {
    auto it = ibn2in_bn_id.find(pair.first);
    if (it != ibn2in_bn_id.end()) {
      CHECK(regst_desc_id2in_bn_id_.emplace(pair.second, it->second).second);
    }
  }

  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
      CHECK(readable_regsts_.emplace(regst_desc_id, std::queue<Regst*>()).second);
      CHECK(regst_desc_id2is_processed_.emplace(regst_desc_id, false).second);
    }
  }
  cur_processed_regst_desc_id_ = -1;
  readable_regst_desc_cnt_ = 0;
  processed_regst_desc_id_cnt_ = 0;
  OF_SET_MSG_HANDLER(&InputWiseCompActor::HandlerNormal);
}

int64_t InputWiseCompActor::ActNumForEachOutput(int64_t regst_desc_id) const {
  return regst_desc_id2in_bn_id_.size();
}

void InputWiseCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  Regst* regst = msg.regst();
  int regst_desc_id = regst->regst_desc_id();
  CHECK(readable_regsts_.find(regst_desc_id) != readable_regsts_.end());
  std::queue<Regst*>& regst_q = readable_regsts_.at(regst_desc_id);
  if (regst_q.empty()) { readable_regst_desc_cnt_ += 1; }
  regst_q.push(regst);
}

bool InputWiseCompActor::IsCustomizedReadReady() {
  CHECK_EQ(-1, cur_processed_regst_desc_id_);
  for (const auto& pair : readable_regsts_) {
    if (pair.second.empty()) { continue; }
    if (regst_desc_id2is_processed_.at(pair.first) == false) {
      cur_processed_regst_desc_id_ = pair.first;
      return true;
    }
  }
  return false;
}

void InputWiseCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  handler(readable_regsts_.at(cur_processed_regst_desc_id_).front());
}

void InputWiseCompActor::Act() {
  std::queue<Regst*>& regst_q = readable_regsts_.at(cur_processed_regst_desc_id_);
  Regst* cur_regst = regst_q.front();

  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  SetKernelCtxOther(&(kernel_ctx.other));
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    CHECK_EQ(cur_processed_regst_desc_id_, regst_desc_id);
    return cur_regst;
  });

  UpdateMemberStatusAfterAct();
  if (NeedSendRegstMsgToConsumer()) {
    AsyncSendRegstMsgToConsumer([&](Regst* regst) {
      regst->set_piece_id(cur_regst->piece_id());
      return true;
    });
    UpdateMemberStatusAfterSendRegstMsgToConsumer();
  }
  AsyncSendRegstMsgToProducer(cur_regst);
}

void InputWiseCompActor::UpdateMemberStatusAfterAct() {
  std::queue<Regst*>& regst_q = readable_regsts_.at(cur_processed_regst_desc_id_);
  regst_q.pop();
  if (regst_q.empty()) { readable_regst_desc_cnt_ -= 1; }
  regst_desc_id2is_processed_.at(cur_processed_regst_desc_id_) = true;
  processed_regst_desc_id_cnt_ += 1;
  just_processed_regst_desc_id_ = cur_processed_regst_desc_id_;
  cur_processed_regst_desc_id_ = -1;
  VirtualUpdateMemberStatusAfterAct();
}

bool InputWiseCompActor::NeedSendRegstMsgToConsumer() {
  return processed_regst_desc_id_cnt_ == regst_desc_id2is_processed_.size();
}

void InputWiseCompActor::UpdateMemberStatusAfterSendRegstMsgToConsumer() {
  for (auto& pair : regst_desc_id2is_processed_) {
    CHECK(pair.second);
    pair.second = false;
  }
  processed_regst_desc_id_cnt_ = 0;
  VirtualUpdateMemberStatusAfterSendRegstMsgToConsumer();
}

void InputWiseCompActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK_EQ(-1, cur_processed_regst_desc_id_);
  CHECK_EQ(0, processed_regst_desc_id_cnt_);
  CHECK_EQ(0, readable_regst_desc_cnt_);
}

bool InputWiseCompActor::ProducedCtrlRegstValid(const Regst* regst) const {
  const RtRegstDesc* rt_regst_desc = regst->regst_desc();
  const RegstDescTypeProto& regst_desc_type_proto = rt_regst_desc->regst_desc_type();
  CHECK(regst_desc_type_proto.has_ctrl_regst_desc());
  const CtrlRegstDesc& ctrl_regst_desc = regst_desc_type_proto.ctrl_regst_desc();
  if (ctrl_regst_desc.has_reliant_regst_desc_id()) {
    if (ctrl_regst_desc.reliant_regst_desc_id() == just_processed_regst_desc_id_) {
      return true;
    } else {
      return false;
    }
  } else {
    return true;
  }
}

}  // namespace oneflow
