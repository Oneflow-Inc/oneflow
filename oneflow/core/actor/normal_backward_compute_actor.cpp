#include "oneflow/core/actor/normal_backward_compute_actor.h"

namespace oneflow {

void NormalBackwardCompActor::VirtualBackwardCompActorInit(
    const TaskProto& task_proto) {
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    if (pair.second == model_regst_desc_id()
        || pair.second == model_tmp_regst_desc_id()) {
      continue;
    }
    (*readable_deq_regsts())[pair.second] = {};
  }
  OF_SET_MSG_HANDLER(&NormalBackwardCompActor::HandlerNormal);
}

int NormalBackwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    if (msg.eord_regst_desc_id() == out_diff_regst_desc_id()) {
      set_is_out_diff_eord(true);
    }
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* cur_regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(cur_regst) != 0) {
      int64_t cur_regst_desc_id = cur_regst->regst_desc_id();
      if (cur_regst_desc_id == model_tmp_regst_desc_id()) {
        CHECK(!model_tmp_regst());
        set_model_tmp_regst(cur_regst);
      } else if (cur_regst_desc_id == model_regst_desc_id()) {
        model_regsts()->push(cur_regst);
      } else {
        auto& rdq = readable_deq_regsts()->at(cur_regst_desc_id);
        if (cur_regst_desc_id == out_diff_regst_desc_id()) {
          HandleOutDiffRegsts(cur_regst, &rdq);
        } else {
          if (cur_regst->col_id() == 0) { rdq.push_back(std::deque<Regst*>()); }
          if (cur_regst_desc_id == out_regst_desc_id() && rdq.size() == 1
              && rdq.front().empty()) {
            AsyncReturnModelRegstUntilMatchCurOutRegst(
                cur_regst->model_version_id());
          }
          rdq.back().push_back(cur_regst);
        }
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

bool NormalBackwardCompActor::IsReadReady() {
  if (model_regst_desc_id() != -1 && model_regsts()->empty()) { return false; }
  if (model_tmp_regst_desc_id() != -1 && !model_tmp_regst()) { return false; }

  auto& out_rdq = readable_deq_regsts()->at(out_regst_desc_id());
  if (out_rdq.empty()) { return false; }
  if (out_rdq.front().empty()) { return false; }

  for (auto& pair : *readable_deq_regsts()) {
    if (pair.first == out_regst_desc_id()) { continue; }
    auto& rdq = pair.second;
    if (rdq.empty()) { return false; }
    if (rdq.front().empty()) { return false; }
    if (order() == ColIdOrder::kDescending) {
      Regst* out_regst = out_rdq.front().back();
      if (out_regst->IsMaxCol()) {
        if (!rdq.front().back()->IsMaxCol()) { return false; }
      } else {
        if (!has_cur_piece_started()) { return false; }
        CHECK(rdq.front().back()->HaveSamePieceColStatusAs(out_regst));
      }
    }
  }
  return true;
}

bool NormalBackwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_out_diff_eord()
         && readable_deq_regsts()->at(out_diff_regst_desc_id()).empty();
}

void NormalBackwardCompActor::CheckBeforeAsyncReturnAllReadableRegst() {
  for (auto& pair : *readable_deq_regsts()) { CHECK(pair.second.empty()); }
}

void NormalBackwardCompActor::Act() {
  auto& out_rdq = readable_deq_regsts()->at(out_regst_desc_id());
  Regst* out_regst = nullptr;
  if (order() == ColIdOrder::kDescending) {
    out_regst = out_rdq.front().back();
    out_rdq.front().pop_back();
  } else {
    out_regst = out_rdq.front().front();
    out_rdq.front().pop_front();
  }
  if (IsLastRegstInPieceWithOrder(out_regst, order())) {
    CHECK(out_rdq.front().empty());
    out_rdq.pop_front();
  }

  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [this](int64_t regst_desc_id) -> Regst* {
                      Regst* regst = GetCurWriteableRegst(regst_desc_id);
                      if (regst == nullptr) {
                        if (regst_desc_id == model_tmp_regst_desc_id()) {
                          return model_tmp_regst();
                        } else if (regst_desc_id == model_regst_desc_id()) {
                          return model_regsts()->front();
                        } else {
                          auto& rdq = readable_deq_regsts()->at(regst_desc_id);
                          if (order() == ColIdOrder::kDescending) {
                            return rdq.front().back();
                          } else {
                            return rdq.front().front();
                          }
                        }
                      } else {
                        return regst;
                      }
                    });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(out_regst->piece_id());
    regst->set_col_id(out_regst->col_id());
    regst->set_max_col_id(out_regst->max_col_id());
    return true;
  });

  if (out_rdq.empty()) {
    AsyncReturnModelRegstUntilLastPieceIdGreaterThan(out_regst->piece_id());
  } else {
    if (!out_rdq.front().empty()) {
      Regst* next_out_regst = nullptr;
      if (order() == ColIdOrder::kDescending) {
        next_out_regst = out_rdq.front().back();
      } else {
        next_out_regst = out_rdq.front().front();
      }
      AsyncReturnModelRegstUntilMatchCurOutRegst(
          next_out_regst->model_version_id());
    }
  }

  for (auto& pair : *readable_deq_regsts()) {
    if (pair.first == out_regst_desc_id()) { continue; }
    auto& rdq = pair.second;
    if (order() == ColIdOrder::kDescending) {
      AsyncSendRegstMsgToProducer(rdq.front().back());
      rdq.front().pop_back();
    } else {
      AsyncSendRegstMsgToProducer(rdq.front().front());
      rdq.front().pop_front();
    }
    if (IsLastRegstInPieceWithOrder(out_regst, order())) {
      CHECK(rdq.front().empty());
      rdq.pop_front();
    }
  }
  if (order() == ColIdOrder::kDescending) {
    if (out_regst->IsMaxCol()) { set_has_cur_piece_started(true); }
    if (out_regst->col_id() == 0) { set_has_cur_piece_started(false); }
  }
  AsyncSendRegstMsgToProducer(out_regst);
}

void NormalBackwardCompActor::ForEachCurReadableRegst(
    std::function<void(const Regst*)> handler) {
  for (const auto& pair : *readable_deq_regsts()) {
    if (order() == ColIdOrder::kDescending) {
      handler(pair.second.front().back());
    } else {
      handler(pair.second.front().front());
    }
  }
  ForCurReadableModelAndModelTmp(handler);
}

REGISTER_ACTOR(TaskType::kNormalBackward, NormalBackwardCompActor);

}  // namespace oneflow
