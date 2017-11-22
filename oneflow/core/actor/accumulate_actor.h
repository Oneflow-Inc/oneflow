#ifndef ONEFLOW_CORE_ACTOR_ACCUMULATE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_ACCUMULATE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class AccumulateActor : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccumulateActor);
  AccumulateActor() = default;
  virtual ~AccumulateActor() = default;

  void Init(const TaskProto&, const ThreadCtx&, int32_t max_acc_cnt);

 private:
  int HandlerNormal(const ActorMsg&) override;
  int HandlerUntilReadAlwaysUnReady(const ActorMsg&);

  bool IsReadReady() override { return !waiting_in_regst_.empty(); }
  void Act() override;

  std::queue<Regst*> waiting_in_regst_;

  void (*MemsetFunc)(DeviceCtx* ctx, void* dst, const char value, size_t sz);

  CudaStreamHandle cuda_handle_;
  int32_t acc_cnt_;
  int32_t max_acc_cnt_;
  int64_t next_acc_piece_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACCUMULATE_ACTOR_H_
