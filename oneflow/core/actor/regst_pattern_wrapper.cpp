#include "oneflow/core/actor/regst_pattern_wrapper.h"

namespace oneflow {

namespace actor {

void NormalPatternWrapper::Init(const TaskProto& task_proto, const ProducedRegstType& produced_regsts, OpActorCtx* ctx) {
  DerivedInit(task_proto);
  consumed_rs_.InitDone();
  produced_rs_.InitDone();

  for (const auto& pair : produced_regsts) {
    if (produced_rs_.HasRegstDescId(pair.first) == false) { continue; }
    for (const auto& regst : pair.second) {
      CHECK_EQ(0, produced_rs_.TryPushBackRegst(regst.get()));
      produced_regst2reading_cnt_.emplace(regst.get(), 0);
    }
  }
  total_reading_cnt_ = 0;
  eord_cnt_ = 0;
  op_actor_ctx_ = ctx;
}

void NormalPatternWrapper::ForEachRegstDescId(std::function<void(int64_t)> handler) const {
  for (const auto& pair : consumed_regst2eord_) { handler(pair.first); }
  // TODO: find produced regst desc ids
  // for (const auto& pair : produced_regst2reading_cnt_) { handler(pair.first); }
}

void NormalPatternWrapper::UpdateWithRegstMsg(const ActorMsg& msg) {
  //TODO(niuchong): not process regst from other machine
  CHECK(msg.SrcMachineId() == Global<MachineCtx>::Get()->this_machine_id());
  Regst* regst = msg.regst();
  bool is_consumed_regst = IsKeyFound(consumed_regst2eord_, regst->regst_desc_id());
  if (is_consumed_regst) {
    UpdateWithConsumedRegstMsg();
  } else {
    UpdateWithProducedRegstMsg();
  }
}

void NormalPatternWrapper::UpdateWithEordMsg(const ActorMsg& msg) {
  consumed_regst2eord_.at(msg.eord_regst_desc_id()) = true;
  eord_cnt_ += 1;
}

void NormalPatternWrapper::IsReady4Act() const {
  return consumed_rs_.IsCurSlotReady() && produced_rs_.IsCurSlotReady();
}

Regst* NormalPatternWrapper::GetRegstByRegstDescId() const {
  int64_t desc_id = regst->regst_desc_id();
  bool is_consumed_regst = IsKeyFound(consumed_regst2eord_, desc_id);
  Regst* regst = is_consumed_regst ? consumed_rs_.Front(desc_id) : produced_rs_.Front(desc_id);
  CHECK_NOTNULL(regst);
  return regst;
}

void NormalPatternWrapper::HandleRegstMsgAfterAct() {
  HandleProducedRegstAfterAct();
  HandleConsumedRegstAfterAct();
}

void CtrlRSWrapper::DerivedInit(const TaskProto& task_proto) {
  for (const auto& pair : task_proto.produced_regst_desc()) {
    if (pair.second.regst_desc_type().has_ctrl_regst_desc()) {
      InsertNewRegstDescId(true, pair.second.regst_desc_id());
    }
  }
  for (const auto& pair : task_proto.produced_regst_desc()) {
    if (pair.first == "in_ctrl") {
      for (int regst_desc_id : pair.second.regst_desc_id()) {
        InsertNewRegstDescId(false, pair.second.regst_desc_id());
      }
    }
  }
}

void CtrlRSWrapper::UpdateWithConsumedRegstMsg(Regst* regst) {
  CHECK_EQ(0, mut_consumed_rs()->TryPushBackRegst(regst));
}

void CtrlRSWrapper::UpdateWithProducedRegstMsg(Regst* regst) {
  CHECK_EQ(0, mut_produced_rs()->TryPushBackRegst(regst));
  UpdateReadingCnt4ProducedRegst(regst, -1);
}

void CtrlRSWrapper::HandleConsumedRegstAfterAct() {
  std::vector<int64_t> regst_desc_ids;
  mut_consumed_rs()->ForEachRegstDeq([&](const std::deque<Regst*>& reg_deq) {
    CHECK(!reg_deq.empty());
    Regst* regst = reg_deq.front();
    int32_t returned_regst_num =
        regst->regst_desc()->regst_desc_type().ctrl_regst_desc().returned_regst_num();
    CHECK_GE(returned_regst_num, 1);
    CHECK_GE(reg_deq.size(), returned_regst_num);
    for (size_t i = 0; i < returned_regst_num; ++i) {
      Regst* regst = reg_deq.at(i);
      // must access regst before sending it to producer
      regst_desc_ids.push_back(regst->regst_desc_id());
      ActorMsgUtil::AsyncSendMsg(op_actor_ctx(),
          ActorMsg::BuildRegstMsgToProducer(op_actor_ctx()->actor_id(), regst->producer_actor_id(), regst));
    }
  });
  mut_consumed_rs()->PopFrontRegsts(regst_desc_ids);
}

void CtrlRSWrapper::HandleProducedRegstAfterAct() {
  std::vector<int64_t> regst_desc_ids;
  mut_produced_rs()->ForEachFrontRegst([&](Regst* regst) {
    regst_desc_ids.push_back(regst->regst_desc_id());
    CHECK_EQ(0, ReadingCnt4ProducedRegst(regst));
    for (int64_t consumer : regst->consumers_actor_id()) {
      ActorMsgUtil::AsyncSendMsg(op_actor_ctx(), ActorMsg::BuildRegstMsgToConsumer(op_actor_ctx()->actor_id(), consumer, regst));
    }
    UpdateReadingCnt4ProducedRegst(regst, regst->consumers_actor_id().size());
  });
  mut_produced_rs()->PopFrontRegsts(regst_desc_ids);
}

void NaiveRSWrapper::DerivedInit(const TaskProto& task_proto) {
  //TODO
}

void InplacePatternWrapper::DerivedInit(const TaskProto& task_proto) {
  //TODO
}

void InplacePatternWrapper::UpdateWithConsumedRegstMsg(Regst* regst) {
  CHECK_EQ(0, mut_consumed_rs()->TryPushBackRegst(regst));
  int64_t corr_out_regst_id = inplace_pair_in2out_.at(regst->regst_desc_id());
  CHECK(regst->packed_blob()->dptr()
        == mut_produced_rs()->Front(corr_out_regst_id)->packed_blob()->dptr());
}

void InplacePatternWrapper::UpdateWithProducedRegstMsg(Regst* regst) {
  CHECK_EQ(0, mut_produced_rs()->TryPushBackRegst(regst));
  UpdateReadingCnt4ProducedRegst(regst, -1);
  if (ReadingCnt4ProducedRegst(regst) == 0) {
    int64_t corr_in_regst_id = inplace_pair_out2in_.at(regst->regst_desc_id());
    Regst* corr_in_regst = mut_consumed_rs()->Front(in_regst_desc_id);
    CHECK_NOTNULL(corr_in_regst);
    ActorMsgUtil::AsyncSendMsg(op_actor_ctx(), ActorMsg::BuildRegstMsgToProducer(op_actor_ctx()->actor_id(), producer, regst));
    CHECK_EQ(mut_consumed_rs()->TryPopFrontRegst(corr_in_regst));
  }
}

void InplacePatternWrapper::HandleConsumedRegstAfterAct() {
  // do noting, cuz it returns consumend regsts only when recv corresponding produced regst
}

void InplacePatternWrapper::HandleProducedRegstAfterAct() {
  std::vector<int64_t> regst_desc_ids;
  mut_produced_rs()->ForEachFrontRegst([&](Regst* regst) {
    regst_desc_ids.push_back(regst->regst_desc_id());
    CHECK_EQ(0, ReadingCnt4ProducedRegst(regst));
    for (int64_t consumer : regst->consumers_actor_id()) {
      ActorMsgUtil::AsyncSendMsg(op_actor_ctx(), ActorMsg::BuildRegstMsgToConsumer(op_actor_ctx()->actor_id(), consumer, regst));
    }
    UpdateReadingCnt4ProducedRegst(regst, regst->consumers_actor_id().size());
  });
  mut_produced_rs()->PopFrontRegsts(regst_desc_ids);
}

}

}
