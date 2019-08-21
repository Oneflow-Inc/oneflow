#include "oneflow/core/actor/op_actor.h"
#include "oneflow/core/actor/regst_handler.h"

namespace oneflow {

namespace actor {

class GeneralOpActor final : public OpActor {
 public:
  void InitMsgHandler() override { OF_SET_OP_ACTOR_MSG_HANDLER(this, &OpActor::HandlerNormal); }
  void InitOtherVal() override {}
};

REGISTER_NEW_ACTOR(TaskType::kLoss, GeneralOpActor);
REGISTER_NEW_ACTOR(TaskType::kOptimizer, GeneralOpActor);

class InplaceOpActor final : public OpActor {
 public:
  void InitMsgHandler() override { OF_SET_OP_ACTOR_MSG_HANDLER(this, &OpActor::HandlerNormal); }
  void InitOtherVal() override {}
};

}  // namespace actor

}  // namespace oneflow
