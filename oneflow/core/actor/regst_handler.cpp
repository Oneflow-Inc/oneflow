#include "oneflow/core/actor/regst_handler.h"
#include "oneflow/core/actor/op_actor.h"

namespace oneflow {

namespace actor {

void NormalRegstHandler::Init(const RegstHandlerProto& handler_proto,
                              const ProducedRegstType& produced_regsts, MsgDeliveryCtx* ctx,
                              std::shared_ptr<void> other) {
  CHECK(type() == handler_proto.type());
  for (int64_t consumed_id : handler_proto.consumed_regst_desc_ids().regst_desc_id()) {
    InsertNewRegstDescId(false, consumed_id);
  }
  for (int64_t produced_id : handler_proto.produced_regst_desc_ids().regst_desc_id()) {
    InsertNewRegstDescId(true, produced_id);
  }
  consumed_rs_.InitedDone();
  produced_rs_.InitedDone();

  for (const auto& pair : produced_regsts) {
    if (produced_rs_.HasRegstDescId(pair.first) == false) { continue; }
    for (const auto& regst : pair.second) {
      CHECK_EQ(0, produced_rs_.TryPushBackRegst(regst.get()));
      produced_regst2reading_cnt_.emplace(regst.get(), 0);
    }
  }
  total_reading_cnt_ = 0;
  eord_cnt_ = 0;
  msg_delivery_ctx_.reset(ctx);
  kernel_other_ = other;
  DerivedInit(handler_proto);
}

void NormalRegstHandler::ForEachRegstDescId(std::function<void(int64_t)> handler) const {
  for (const auto& pair : consumed_regst2eord_) { handler(pair.first); }
  // TODO: find produced regst desc ids
  // for (const auto& pair : produced_regst2reading_cnt_) { handler(pair.first); }
}

void NormalRegstHandler::UpdateWithRegstMsg(const ActorMsg& msg) {
  // TODO(niuchong): not process regst from other machine
  CHECK(msg.SrcMachineId() == Global<MachineCtx>::Get()->this_machine_id());
  Regst* regst = msg.regst();
  bool is_consumed_regst = IsKeyFound(consumed_regst2eord_, regst->regst_desc_id());
  if (is_consumed_regst) {
    UpdateWithConsumedRegstMsg(msg);
  } else {
    UpdateWithProducedRegstMsg(msg);
  }
}

void NormalRegstHandler::UpdateWithEordMsg(const ActorMsg& msg) {
  consumed_regst2eord_.at(msg.eord_regst_desc_id()) = true;
  eord_cnt_ += 1;
}

bool NormalRegstHandler::IsReady() const {
  return consumed_rs_.IsCurSlotReady() && produced_rs_.IsCurSlotReady();
}

Regst* NormalRegstHandler::GetRegstByRegstDescId(int64_t desc_id) const {
  bool is_consumed_regst = IsKeyFound(consumed_regst2eord_, desc_id);
  Regst* regst = is_consumed_regst ? consumed_rs_.Front(desc_id) : produced_rs_.Front(desc_id);
  CHECK_NOTNULL(regst);
  return regst;
}

void NormalRegstHandler::HandleRegstMsgAfterAct() {
  HandleProducedRegstAfterAct();
  HandleConsumedRegstAfterAct();
}

void CtrlRegstHandler::UpdateWithConsumedRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, mut_consumed_rs()->TryPushBackRegst(msg.regst()));
}

void CtrlRegstHandler::UpdateWithProducedRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, mut_produced_rs()->TryPushBackRegst(msg.regst()));
  UpdateReadingCnt4ProducedRegst(msg.regst(), -1);
}

void CtrlRegstHandler::HandleConsumedRegstAfterAct() {
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
      ActorMsgUtil::AsyncSendMsg(
          msg_delivery_ctx(), ActorMsg::BuildRegstMsgToProducer(msg_delivery_ctx()->actor_id,
                                                                regst->producer_actor_id(), regst));
    }
  });
  mut_consumed_rs()->PopFrontRegsts(regst_desc_ids);
}

void CtrlRegstHandler::HandleProducedRegstAfterAct() {
  std::vector<int64_t> regst_desc_ids;
  mut_produced_rs()->ForEachFrontRegst([&](Regst* regst) {
    regst_desc_ids.push_back(regst->regst_desc_id());
    CHECK_EQ(0, ReadingCnt4ProducedRegst(regst));
    for (int64_t consumer : regst->consumers_actor_id()) {
      ActorMsgUtil::AsyncSendMsg(
          msg_delivery_ctx(),
          ActorMsg::BuildRegstMsgToConsumer(msg_delivery_ctx()->actor_id, consumer, regst));
    }
    UpdateReadingCnt4ProducedRegst(regst, regst->consumers_actor_id().size());
  });
  mut_produced_rs()->PopFrontRegsts(regst_desc_ids);
}

void NaiveRegstHandler::UpdateWithConsumedRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, mut_consumed_rs()->TryPushBackRegst(msg.regst()));
}

void NaiveRegstHandler::UpdateWithProducedRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, mut_produced_rs()->TryPushBackRegst(msg.regst()));
  UpdateReadingCnt4ProducedRegst(msg.regst(), -1);
}

void NaiveRegstHandler::HandleConsumedRegstAfterAct() {
  std::vector<int64_t> regst_desc_ids;
  mut_consumed_rs()->ForEachFrontRegst([&](Regst* regst) {
    // must access regst before sending it to producer
    regst_desc_ids.push_back(regst->regst_desc_id());
    ActorMsgUtil::AsyncSendMsg(
        msg_delivery_ctx(), ActorMsg::BuildRegstMsgToProducer(msg_delivery_ctx()->actor_id,
                                                              regst->producer_actor_id(), regst));
  });
  mut_consumed_rs()->PopFrontRegsts(regst_desc_ids);
}

void NaiveRegstHandler::HandleProducedRegstAfterAct() {
  std::vector<int64_t> regst_desc_ids;
  mut_produced_rs()->ForEachFrontRegst([&](Regst* regst) {
    regst_desc_ids.push_back(regst->regst_desc_id());
    CHECK_EQ(0, ReadingCnt4ProducedRegst(regst));
    for (int64_t consumer : regst->consumers_actor_id()) {
      ActorMsgUtil::AsyncSendMsg(
          msg_delivery_ctx(),
          ActorMsg::BuildRegstMsgToConsumer(msg_delivery_ctx()->actor_id, consumer, regst));
    }
    UpdateReadingCnt4ProducedRegst(regst, regst->consumers_actor_id().size());
  });
  mut_produced_rs()->PopFrontRegsts(regst_desc_ids);
}
void InplaceRegstHandler::DerivedInit(const RegstHandlerProto& handler_proto) {
  CHECK(handler_proto.further_desc_case() == RegstHandlerProto::kInplaceDesc);
  for (const auto& pair : handler_proto.inplace_desc().paired_out2in()) {
    inplace_pair_out2in_.emplace(pair.first, pair.second);
    inplace_pair_in2out_.emplace(pair.second, pair.first);
  }
}

void InplaceRegstHandler::UpdateWithConsumedRegstMsg(const ActorMsg& msg) {
  Regst* regst = msg.regst();
  CHECK_EQ(0, mut_consumed_rs()->TryPushBackRegst(regst));
  int64_t corr_out_regst_id = inplace_pair_in2out_.at(regst->regst_desc_id());
  CHECK(regst->packed_blob()->dptr()
        == mut_produced_rs()->Front(corr_out_regst_id)->packed_blob()->dptr());
}

void InplaceRegstHandler::UpdateWithProducedRegstMsg(const ActorMsg& msg) {
  Regst* regst = msg.regst();
  CHECK_EQ(0, mut_produced_rs()->TryPushBackRegst(regst));
  UpdateReadingCnt4ProducedRegst(regst, -1);
  if (ReadingCnt4ProducedRegst(regst) == 0) {
    int64_t corr_in_regst_id = inplace_pair_out2in_.at(regst->regst_desc_id());
    Regst* corr_in_regst = mut_consumed_rs()->Front(corr_in_regst_id);
    CHECK_NOTNULL(corr_in_regst);
    ActorMsgUtil::AsyncSendMsg(
        msg_delivery_ctx(), ActorMsg::BuildRegstMsgToProducer(msg_delivery_ctx()->actor_id,
                                                              regst->producer_actor_id(), regst));
    CHECK_EQ(0, mut_consumed_rs()->TryPopFrontRegst(corr_in_regst_id));
  }
}

void InplaceRegstHandler::HandleConsumedRegstAfterAct() {
  // do noting, cuz it returns consumend regsts only when recv corresponding produced regst
}

void InplaceRegstHandler::HandleProducedRegstAfterAct() {
  std::vector<int64_t> regst_desc_ids;
  mut_produced_rs()->ForEachFrontRegst([&](Regst* regst) {
    regst_desc_ids.push_back(regst->regst_desc_id());
    CHECK_EQ(0, ReadingCnt4ProducedRegst(regst));
    for (int64_t consumer : regst->consumers_actor_id()) {
      ActorMsgUtil::AsyncSendMsg(
          msg_delivery_ctx(),
          ActorMsg::BuildRegstMsgToConsumer(msg_delivery_ctx()->actor_id, consumer, regst));
    }
    UpdateReadingCnt4ProducedRegst(regst, regst->consumers_actor_id().size());
  });
  mut_produced_rs()->PopFrontRegsts(regst_desc_ids);
}

}  // namespace actor

}  // namespace oneflow
