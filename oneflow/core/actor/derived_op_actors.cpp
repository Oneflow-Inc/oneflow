#include "oneflow/core/actor/op_actor.h"
#include "oneflow/core/actor/regst_handler.h"

namespace oneflow {

namespace actor {

class GeneralOpActor final : public OpActor {
 public:
  void InitMsgHandler() override { OF_SET_OP_ACTOR_MSG_HANDLER(&OpActor::HandlerNormal); }
  void VirtualSetRegstHandlers() override { InsertRegstHandler(new NaiveRegstHandler); }
  void InitOtherVal() override {}
};

REGISTER_NEW_ACTOR(TaskType::kLoss, GeneralOpActor);

class InplaceOpActor final : public OpActor {
 public:
  void InitMsgHandler() override { OF_SET_OP_ACTOR_MSG_HANDLER(&OpActor::HandlerNormal); }
  void VirtualSetRegstHandlers() override {
    InsertRegstHandler(new NaiveRegstHandler);
    InsertRegstHandler(new InplaceRegstHandler);
  }
  void InitOtherVal() override {}
};

}  // namespace actor

}  // namespace oneflow
