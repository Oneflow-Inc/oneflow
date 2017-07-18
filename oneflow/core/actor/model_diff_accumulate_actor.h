#ifndef ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_DIFF_ACCUMULATE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class MdDiffAccActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdDiffAccActor);
  MdDiffAccActor() = default;
  ~MdDiffAccActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;
  int HandlerWaitUntilNoReadableRegst(const ActorMsg&) override;

  bool IsReadReady() override { return !waiting_in_regst_.empty(); }
  void Act() override;

  std::queue<std::shared_ptr<RegstWrapper>> waiting_in_regst_;

  void (*MemsetFunc)(const KernelCtx& ctx, void* dst, const char value,
                     size_t sz);

  CudaStreamHandle cuda_handle_;
  HashMap<Regst*, int32_t> model_diff_acc_cnt_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COPY_HD_ACTOR_H_
