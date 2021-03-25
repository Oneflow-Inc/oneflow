/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_ACTOR_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_ACTOR_H_

#include "oneflow/core/actor/act_event.pb.h"
#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/thread/thread_context.h"
#include "oneflow/core/actor/register_slot.h"

namespace oneflow {

class Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Actor);
  virtual ~Actor() = default;

  const JobDesc& job_desc() const { return *job_desc_; }

  void Init(const JobDesc* job_desc, const TaskProto&, const ThreadCtx&);

  // 1: success, and actor finish
  // 0: success, and actor not finish
  int ProcessMsg(const ActorMsg& msg) { return (this->*msg_handler_)(msg); }

  int64_t machine_id() const { return Global<IDMgr>::Get()->MachineId4ActorId(actor_id_); }
  int64_t thrd_id() const { return Global<IDMgr>::Get()->ThrdId4ActorId(actor_id_); }
  int64_t actor_id() const { return actor_id_; }

 protected:
  struct BlobInfo {
    LogicalBlobId lbi;
    int64_t regst_desc_id;
    int64_t ordinal;
    RegstSlot* rs;
  };
  struct ExecKernel {
    std::unique_ptr<const Kernel> kernel;
    HashMap<std::string, BlobInfo> bn_in_op2blob_info;
  };
  using MsgHandler = int (Actor::*)(const ActorMsg&);
  enum class RegstNameType { kNaive = 0, kCustomized };

  // Util
  Actor() = default;
  const ParallelContext* parallel_ctx() const { return parallel_ctx_.get(); }
  bool ReceiveAllEordMsg() const { return remaining_eord_cnt_ == 0; }
  bool ReceiveEordMsg(int64_t regst_desc_id) const;
  DeviceType GetDeviceType() const;
  virtual void VirtualActorInit(const TaskProto&) {}
  int64_t Name2SoleRegstDescId(const std::string& name) const;
  const std::vector<int64_t>& Name2RegstDescIds(const std::string& name) const;
  virtual void InitDeviceCtx(const ThreadCtx&);
  std::unique_ptr<DeviceCtx>& mut_device_ctx() { return device_ctx_; }
  KernelCtx GenDefaultKernelCtx() const;
  const std::vector<ExecKernel>& exec_kernel_vec() { return exec_kernel_vec_; }
  virtual void SetReadableRegstInfo(const Regst*, ReadableRegstInfo*) const;
  void ForEachCurNaiveReadableDataRegst(std::function<void(const Regst*)>) const;

  int64_t act_id() const { return act_id_; }
  int64_t ReadingCnt4ProducedRegst(Regst* regst) const;
  void IncreaseReadingCnt4ProducedRegst(Regst* regst, int64_t val);
  void IncreaseTotalReadingCnt(int64_t val) { total_reading_cnt_ += val; }
  int64_t GetPieceId4NaiveCurReadableDataRegst() const;
  int64_t GetPieceId4NaiveOrInplaceCurReadableDataRegst() const;

  // Msg Handler
  void set_msg_handler(MsgHandler val) { msg_handler_ = val; }
#define OF_SET_MSG_HANDLER(val)                                   \
  do {                                                            \
    LOG(INFO) << "actor " << actor_id() << " switch to " << #val; \
    set_msg_handler(static_cast<MsgHandler>(val));                \
  } while (0)

  // Common Handlers and related virtual method
  int HandlerNormal(const ActorMsg& msg);
  int HandlerZombie(const ActorMsg& msg);

  virtual bool ConsumedCtrlRegstValid(int64_t regst_desc_id) const { return true; }
  virtual bool ProducedCtrlRegstValid(int64_t regst_desc_id) const { return true; }

  // Async Do on device_ctx_
  void AsyncDo(std::function<void()> func) { device_ctx_->AddCallBack(func); }
  void AsyncLaunchKernel(const KernelCtx&, std::function<Regst*(int64_t)> Regst4RegstDescId);
  void AsyncLaunchKernel(const KernelCtx&);

  // Util For Derived Actor to Send Msg
  void EnqueueAsyncMsg(const ActorMsg&);
  void HandleProducedNaiveDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess,
                                              std::function<bool(int64_t)> IsAllowedActor);
  void HandleProducedNaiveDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess);
  void HandleProducedNaiveDataRegstToConsumer(std::function<bool(int64_t)> IsAllowedActor);
  void HandleProducedNaiveDataRegstToConsumer();
  void HandleProducedInplaceDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess,
                                                std::function<bool(int64_t)> IsAllowedActor);
  void HandleProducedInplaceDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess);
  void HandleProducedInplaceDataRegstToConsumer(std::function<bool(int64_t)> IsAllowedActor);
  void HandleProducedInplaceDataRegstToConsumer();
  void AsyncSendRegstMsgToConsumer(Regst* regst);
  void AsyncSendRegstMsgToConsumer(Regst* regst, std::function<bool(int64_t)> IsAllowedActor);

  void HandleConsumedNaiveDataRegstToProducer(std::function<bool(Regst*)> IsAllowedRegst);
  void AsyncSendRegstMsgToProducer(Regst*);
  void AsyncSendRegstMsgToProducer(Regst*, int64_t producer);
  void AsyncSendEORDMsgForAllProducedRegstDesc();
  void AsyncSendQueuedMsg();

  // Get Regst
  Regst* GetNaiveCurReadable(int64_t regst_desc_id) const;
  Regst* GetNaiveCurReadable(const std::string& name) const {
    return GetNaiveCurReadable(Name2SoleRegstDescId(name));
  }
  Regst* GetNaiveOrInplaceCurReadable(int64_t regst_desc_id) const;
  Regst* GetNaiveOrInplaceCurReadable(const std::string& name) const {
    return GetNaiveOrInplaceCurReadable(Name2SoleRegstDescId(name));
  }
  Regst* GetNaiveCurWriteable(int64_t regst_desc_id) const;
  Regst* GetNaiveCurWriteable(const std::string& name) const {
    return GetNaiveCurWriteable(Name2SoleRegstDescId(name));
  }
  Regst* GetNaiveOrInplaceCurWriteable(int64_t regst_desc_id) const;
  Regst* GetNaiveOrInplaceCurWriteable(const std::string& name) const {
    return GetNaiveOrInplaceCurWriteable(Name2SoleRegstDescId(name));
  }
  Regst* GetSoleProducedRegst4RegstDescId(int64_t regst_desc_id) const;
  void ForEachProducedRegst(const std::function<void(Regst*)>&) const;
  int64_t HandleRegstToConsumer(Regst* regst, std::function<bool(int64_t)> IsAllowedActor);

 protected:
  int64_t GetGlobalWorkStreamId() const;
  virtual bool NeedCollectActEvent() const {
    return Global<RuntimeCtx>::Get()->NeedCollectActEvent();
  }
  bool IsConsumedCtrlRegstDescId(int64_t regst_desc_id) {
    return consumed_ctrl_regst_desc_ids_.find(regst_desc_id) != consumed_ctrl_regst_desc_ids_.end();
  }
  bool IsProducedCtrlRegstDescId(int64_t regst_desc_id) {
    return produced_ctrl_regst_desc_ids_.find(regst_desc_id) != produced_ctrl_regst_desc_ids_.end();
  }

  // Process Msg
  virtual void NormalProcessNaiveReadableDataRegstMsg(const std::deque<Regst*>&) {}
  virtual bool NormalTryProcessReadableMsgFromOtherMachine(const ActorMsg&) { return false; }
  int TryUpdtStateAsProducedRegst(Regst* regst);

  // Act
  void ActUntilFail();
  virtual void Act() { UNIMPLEMENTED(); }
  virtual int64_t ActNumForEachOutput(int64_t regst_desc_id) const { return 1; }
  virtual bool CheckOutputActId(int64_t regst_desc_id) const {
    return true;  // TODO(jiyuan): figure out the ActNumForEachOutput of the model regsts to MdSave
                  // area
  }
  void TryLogActEvent(const std::function<void()>& Callback) const;

  // Ready
  bool IsReadReady() const;
  bool IsWriteReady() const;

  // Naive, Inplace Or Customized
  virtual void TakeOverInplaceConsumedAndProduced(
      const PbMap<std::string, RegstDescProto>& produced_ids);
  void TakeOverNaiveConsumed(const PbMap<std::string, RegstDescIdSet>& consumed_ids);
  void TakeOverNaiveProduced(const PbMap<std::string, RegstDescProto>& produced_ids);
  void InitBnInOp2BlobInfo(const TaskProto& task_proto);

  // Send Msgs
  void AsyncSendNaiveProducedRegstMsgToConsumer();
  virtual void VirtualAsyncSendNaiveProducedRegstMsgToConsumer();
  virtual void VirtualAsyncSendInplaceProducedRegstMsgToConsumer();
  void AsyncSendInplaceProducedRegstMsgToConsumer();
  void AsyncSendNaiveConsumedRegstMsgToProducer();
  virtual void VirtualAsyncSendNaiveConsumedRegstMsgToProducer();
  void AsyncSendConsumedCtrlRegstMsgToProducer();
  void AsyncSendProducedCtrlRegstMsgToConsumer();

  // Customized Consumed virtual func
  virtual void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)>) const {}
  virtual void NormalProcessCustomizedEordMsg(const ActorMsg&) {}
  virtual void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) { UNIMPLEMENTED(); }
  virtual bool IsCustomizedReadReady() const { return true; }
  virtual bool IsCustomizedReadAlwaysUnReadyFromNow() const { return false; }
  virtual std::pair<RegstNameType, HashSet<std::string>>
  GetNaiveOrCustomizedConsumedRegstDescName() {
    return std::make_pair(RegstNameType::kCustomized, HashSet<std::string>{});
  }
  virtual void AsyncSendCustomizedProducedRegstMsgToConsumer() {}
  virtual void AsyncReturnAllCustomizedReadableRegst() {}

  // Customized Produced virtual func
  virtual void UpdtStateAsCustomizedProducedRegst(Regst* regst) { UNIMPLEMENTED(); }
  virtual bool IsCustomizedWriteReady() const { return true; }
  virtual std::pair<RegstNameType, HashSet<std::string>>
  GetNaiveOrCustomizedProducedRegstDescName() {
    return std::make_pair(RegstNameType::kCustomized, HashSet<std::string>{});
  }
  virtual void AsyncSendCustomizedConsumedRegstMsgToProducer() {}
  void AsyncRetInplaceConsumedRegstIfNoConsumer();

  const JobDesc* job_desc_;
  int64_t actor_id_;
  int64_t act_id_;
  std::unique_ptr<ParallelContext> parallel_ctx_;
  std::vector<ExecKernel> exec_kernel_vec_;
  HashMap<std::string, std::vector<int64_t>> name2regst_desc_id_;
  MsgHandler msg_handler_;
  std::unique_ptr<DeviceCtx> device_ctx_;
  HashSet<int64_t> eord_regst_desc_ids_;
  int64_t remaining_eord_cnt_;

  HashMap<int64_t, std::vector<std::unique_ptr<Regst>>> produced_regsts_;
  HashMap<int64_t, int64_t> produced_regst2expected_act_id_;
  HashMap<Regst*, int64_t> produced_regst2reading_cnt_;
  int64_t total_reading_cnt_;

  RegstSlot naive_produced_rs_;
  RegstSlot naive_consumed_rs_;
  bool is_naive_consumed_eord_;

  HashSet<int64_t> produced_ctrl_regst_desc_ids_;
  HashSet<int64_t> consumed_ctrl_regst_desc_ids_;

  RegstSlot inplace_consumed_rs_;
  RegstSlot inplace_produced_rs_;
  bool is_inplace_consumed_eord_;
  HashSet<int64_t> inplace_in_ids_with_no_out_consumed_;
  HashMap<int64_t, int64_t> inplace_regst_desc_id_in2out_;
  HashMap<int64_t, int64_t> inplace_regst_desc_id_out2in_;

  std::deque<ActorMsg> async_msg_queue_;
  bool is_kernel_launch_synchronized_;
  std::vector<int64_t> tmp_regst_desc_id_vec_;
};

std::unique_ptr<Actor> NewActor(const TaskProto&, const ThreadCtx&);

#define REGISTER_ACTOR(task_type, ActorType) REGISTER_CLASS(int32_t, task_type, Actor, ActorType)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACTOR_H_
