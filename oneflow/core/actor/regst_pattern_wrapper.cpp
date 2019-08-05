#include "oneflow/core/actor/regst_pattern_wrapper.h"

namespace oneflow {

namespace actor {

void NormalPatternWrapper::Init(const TaskProto& task_proto) {
  DerivedInit();
  eord_cnt_ = 0;
  total_reading_cnt_ = 0;
  consumed_rs_.InitDone();
  produced_rs_.InitDone();
}

void NormalPatternWrapper::ForEachRegstDescId(std::function<void(int64_t)> handler) const {
  for (const auto& pair : consumed_regst2eord_) { handler(pair.first); }
  for (const auto& pair : produced_regst2reading_cnt_) { handler(pair.first); }
}

void NormalPatternWrapper::UpdateWithRegstMsg(const ActorMsg& msg) {
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
