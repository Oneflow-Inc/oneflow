#include "oneflow/core/actor/op_actor.h"
#include "oneflow/core/actor/regst_handler.h"

namespace oneflow {

namespace actor {

class GeneralOpActor final : public OpActor {
 public:
  void InitMsgHandler() override {
    set_initial_msg_handler(std::bind(&OpActor::HandlerNormal, this, std::placeholders::_1));
  }
  void VirtualSetRegstHandlers() override { InsertRegstHandler(new NaiveRegstHandler); }
  void InitOtherVal() override {}
};

REGISTER_NEW_ACTOR(TaskType::kLoss, GeneralOpActor);

class InplaceOpActor final : public OpActor {
 public:
  void InitMsgHandler() override {
    set_initial_msg_handler(std::bind(&OpActor::HandlerNormal, this, std::placeholders::_1));
  }
  void VirtualSetRegstHandlers() override {
    InsertRegstHandler(new NaiveRegstHandler);
    InsertRegstHandler(new InplaceRegstHandler);
  }
  void InitOtherVal() override {}
};

}  // namespace actor

}  // namespace oneflow
