#include "oneflow/core/actor/regst_pattern_wrapper.h"

namespace oneflow {

namespace actor {

void NormalPatternWrapper::Init(const TaskProto& task_proto, const ProducedRegstType& produced_regsts) {
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
  HandleConsumedRegstAfterAct();
  HandleProducedRegstAfterAct();
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
  size_t& reading_cnt = produced_regst2reading_cnt_.at(regst);
  reading_cnt -= 1;
  total_reading_cnt_ -= 1;
}

void NaiveRSWrapper::DerivedInit(const TaskProto& task_proto) {
  for (const auto& pair : task_proto.produced_regst_desc()) {
    if (pair.second.regst_desc_type().has_naive_regst_desc()) {
      InsertNewRegstDescId(true, pair.second.regst_desc_id());
    }
  }
  for (const auto& pair : task_proto.produced_regst_desc()) {
    if (pair.second.regst_desc_type().has_naive_regst_desc()) {
      InsertNewRegstDescId(false, pair.second.regst_desc_id());
    }
  }
}

}

}
