#ifndef ONEFLOW_CORE_ACTOR_SINK_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_SINK_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class SinkCompActor : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SinkCompActor);
  SinkCompActor() = default;
  virtual ~SinkCompActor() = default;

 protected:
  virtual void VirtualSinkCompActorInit(const TaskProto&) {}
  virtual KernelCtx GenSinkKernelCtx() { return GenDefaultKernelCtx(); }

 private:
  void VirtualCompActorInit(const TaskProto&) override;

  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override { return in_regst_; }
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;

  std::list<RegstEvent> CurActComsumedRegstEvents() const override;

  Regst* in_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_SINK_COMPUTE_ACTOR_H_
