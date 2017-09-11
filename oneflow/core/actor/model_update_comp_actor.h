#ifndef ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class MdUpdtCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtCompActor);
  MdUpdtCompActor() = default;
  ~MdUpdtCompActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;

 private:
  int HandlerBeforeInitDeviceCtx(const ActorMsg&);
  int HandlerBeforeInitializeModel(const ActorMsg&);
  int HandlerBeforeSendInitialModel(const ActorMsg&);
  int HandlerNormal(const ActorMsg&) override;
  int HandlerWaitUntilNoReadableRegst(const ActorMsg&) override;

  bool IsWriteReady() const override {
    return Actor::IsWriteReady()
           && CurWriteableRegstNum4DescId(model_regst_desc_id_) >= 2;
  }
  bool IsReadReady() override { return !waiting_model_diff_acc_queue_.empty(); }
  void Act() override;
  void AsyncCopyModelFromCurToNext() {
    Regst* model_regst = GetCurWriteableRegst(model_regst_desc_id_);
    Regst* next_model_regst = GetNextWriteableRegst(model_regst_desc_id_);
    MemcpyFunc(GenDefaultKernelCtx().device_ctx,
               next_model_regst->packed_blob()->mut_dptr(),
               model_regst->packed_blob()->dptr(),
               next_model_regst->packed_blob()->TotalByteSize());
  }

  CudaStreamHandle cuda_handle_;
  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  int64_t data_tmp_regst_desc_id_;
  std::queue<Regst*> waiting_model_diff_acc_queue_;
  int64_t next_model_version_id_;
  int64_t related_save_task_id_;
  uint32_t random_seed_;

  std::function<void(DeviceCtx*, void*, const void*, size_t)> MemcpyFunc;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_
