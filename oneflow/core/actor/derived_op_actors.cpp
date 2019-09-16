#include "oneflow/core/actor/op_actor.h"
#include "oneflow/core/actor/regst_handler.h"
#include "oneflow/core/kernel/decode_ofrecord_kernel.h"

namespace oneflow {

namespace actor {

class GeneralOpActor final : public OpActor {
 public:
  void InitMsgHandler() override { OF_SET_OP_ACTOR_MSG_HANDLER(this, &OpActor::HandlerNormal); }
  void InitOtherVal() override {}
  void SetOtherVal4CurAct(void*) override {}
};

REGISTER_NEW_ACTOR(TaskType::kLoss, GeneralOpActor);
REGISTER_NEW_ACTOR(TaskType::kNormalForward, GeneralOpActor);
REGISTER_NEW_ACTOR(TaskType::kCopyHd, GeneralOpActor);

class OptimizerOpActor final : public OpActor {
  void InitMsgHandler() override { OF_SET_OP_ACTOR_MSG_HANDLER(this, &OpActor::HandlerNormal); }
  void InitOtherVal() override {
    std::shared_ptr<void> cur_batch_num = std::make_shared<int64_t>(0);
    set_other_val(cur_batch_num);
  }
  void SetOtherVal4CurAct(void* other) override {
    int64_t* cur_batch_num = static_cast<int64_t*>(other);
    *cur_batch_num = *cur_batch_num + 1;
  }
};
REGISTER_NEW_ACTOR(TaskType::kOptimizer, OptimizerOpActor);

class InplaceOpActor final : public OpActor {
 public:
  void InitMsgHandler() override { OF_SET_OP_ACTOR_MSG_HANDLER(this, &OpActor::HandlerNormal); }
  void InitOtherVal() override {}
  void SetOtherVal4CurAct(void*) override {}
};

class SourceOpActor final : public OpActor {
 public:
  static int HandlerWaitToStart(OpActor* actor, const ActorMsg& msg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
    OF_SET_OP_ACTOR_MSG_HANDLER(dynamic_cast<SourceOpActor*>(actor), &OpActor::HandlerNormal);
    return actor->ProcessMsg(msg);
  }
  void InitMsgHandler() override {
    OF_SET_OP_ACTOR_MSG_HANDLER(this, &SourceOpActor::HandlerWaitToStart);
  }
  void InitOtherVal() override {}
  void SetOtherVal4CurAct(void*) override {}
};
REGISTER_NEW_ACTOR(TaskType::kDecodeRandom, SourceOpActor);

class DecodeOpActor final : public OpActor {
 public:
  void InitMsgHandler() override { OF_SET_OP_ACTOR_MSG_HANDLER(this, &OpActor::HandlerNormal); }
  void InitOtherVal() override {
    std::shared_ptr<DecodeStatus> decode_status = std::make_shared<DecodeStatus>();
    decode_status->cur_col_id_ = 0;
    decode_status->max_col_id_ = 0;
    set_other_val(std::shared_ptr<void>(decode_status));
  }
  void SetOtherVal4CurAct(void* other) override {
    DecodeStatus* decode_status = static_cast<DecodeStatus*>(other);
    if (decode_status->cur_col_id_ == decode_status->max_col_id_) {
      decode_status->cur_col_id_ = 0;
      decode_status->max_col_id_ = 0;
    } else {
      (decode_status->cur_col_id_) += 1;
    }
  }
};
REGISTER_NEW_ACTOR(TaskType::kDecode, DecodeOpActor);

}  // namespace actor

}  // namespace oneflow
