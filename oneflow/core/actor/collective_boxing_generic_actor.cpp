#include "oneflow/core/actor/actor.h"
#include "oneflow/core/device/collective_boxing_device_context.h"

namespace oneflow {

class CollectiveBoxingGenericActor : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingGenericActor);
  CollectiveBoxingGenericActor() = default;
  ~CollectiveBoxingGenericActor() override = default;

 private:
  void Act() override { AsyncLaunchKernel(GenDefaultKernelCtx()); }

  void VirtualActorInit(const TaskProto&) override {
    piece_id_ = 0;
    OF_SET_MSG_HANDLER(&CollectiveBoxingGenericActor::HandlerNormal);
  }

  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override {
    HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
      regst->set_piece_id(piece_id_);
      return true;
    });
    piece_id_ += 1;
  }

  void InitDeviceCtx(const ThreadCtx& thread_ctx) override {
    mut_device_ctx().reset(new CollectiveBoxingDeviceCtx());
  }

  int64_t piece_id_ = 0;
};

REGISTER_ACTOR(TaskType::kCollectiveBoxingGeneric, CollectiveBoxingGenericActor);

}  // namespace oneflow
