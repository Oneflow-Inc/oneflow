#include "oneflow/core/actor/op_actor.h"
#include "oneflow/core/actor/regst_handler.h"

namespace oneflow {

namespace actor {

class GeneralOpActor final : public OpActor {
 public:
  void InitMsgHandler() override { OF_SET_OP_ACTOR_MSG_HANDLER(this, &OpActor::HandlerNormal); }
  void InitOtherVal() override {}
  void SetOtherVal4CurAct(void*) override {}
};

REGISTER_NEW_ACTOR(TaskType::kLoss, GeneralOpActor);

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

}  // namespace actor

}  // namespace oneflow
