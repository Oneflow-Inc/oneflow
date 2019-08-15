#include "oneflow/core/actor/op_actor.h"

namespace oneflow {

namespace actor {

class GeneralOpActor final : public OpActor {
 public:
  void InitMsgHandler() override {}
  void InitRegstHandlers() override;
  void InitOtherVal() override;
};

}  // namespace actor

}  // namespace oneflow
