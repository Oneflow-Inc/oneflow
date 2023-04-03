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
#ifndef ONEFLOW_CORE_LAZY_ACTOR_ACTOR_H_
#define ONEFLOW_CORE_LAZY_ACTOR_ACTOR_H_

#include "oneflow/core/lazy/actor/actor_base.h"
#include "oneflow/core/lazy/actor/actor_message_bus.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/lazy/actor/register_slot.h"

namespace oneflow {

class Actor : public ActorBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Actor);
  virtual ~Actor();

  void Init(const JobDesc* job_desc, ActorContext* actor_ctx) override;

  // 1: success, and actor finish
  // 0: success, and actor not finish
  int ProcessMsg(const ActorMsg& msg) override { return (this->*msg_handler_)(msg); }

  int64_t machine_id() const { return MachineId4ActorId(actor_id_); }
  int64_t actor_id() const { return actor_id_; }
  int64_t job_id() const { return job_id_; }

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
    std::unique_ptr<KernelContext> kernel_ctx;
  };
  using MsgHandler = int (Actor::*)(const ActorMsg&);
  enum class RegstNameType { kNaive = 0, kCustomized };

  // Util
  Actor() = default;
  bool ReceiveAllEordMsg() const { return remaining_eord_cnt_ == 0; }
  bool ReceiveEordMsg(int64_t regst_desc_id) const;
  virtual void VirtualActorInit(const TaskProto&) {}
  int64_t Name2SoleRegstDescId(const std::string& name) const;
  const std::vector<int64_t>& Name2RegstDescIds(const std::string& name) const;
  ActorContext* actor_ctx() const { return actor_ctx_; }
  const std::vector<ExecKernel>& exec_kernel_vec() { return exec_kernel_vec_; }
  void ForEachCurNaiveReadableDataRegst(const std::function<void(const Regst*)>&) const;

  int64_t ReadingCnt4ProducedRegst(Regst* regst) const;
  void IncreaseReadingCnt4ProducedRegst(Regst* regst, int64_t val);
  void IncreaseTotalReadingCnt(int64_t val) { total_reading_cnt_ += val; }

  // Msg Handler
  void set_msg_handler(MsgHandler val) { msg_handler_ = val; }
#define OF_SET_MSG_HANDLER(val)                                 \
  do {                                                          \
    VLOG(3) << "actor " << actor_id() << " switch to " << #val; \
    set_msg_handler(static_cast<MsgHandler>(val));              \
  } while (0)

  // Common Handlers and related virtual method
  int HandlerNormal(const ActorMsg& msg);
  int HandlerZombie(const ActorMsg& msg);

  virtual bool ConsumedCtrlRegstValid(int64_t regst_desc_id) const { return true; }
  virtual bool ProducedCtrlRegstValid(int64_t regst_desc_id) const { return true; }

  void AsyncLaunchKernel(std::function<Regst*(int64_t)> Regst4RegstDescId);
  void AsyncLaunchKernel();

  // Util For Derived Actor to Send Msg
  void EnqueueAsyncMsg(const ActorMsg&);
  void HandleProducedNaiveDataRegstToConsumer();
  void PrepareProducedNaiveInplaceDataRegst();
  void HandleProducedInplaceDataRegstToConsumer();

  void HandleConsumedNaiveDataRegstToProducer();
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
  int64_t HandleRegstToConsumer(Regst* regst);

 protected:
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

  virtual void AddCallback(std::function<void()> callback);

  int64_t actor_id_;
  int64_t thrd_id_;
  int64_t job_id_;
  std::vector<ExecKernel> exec_kernel_vec_;
  HashMap<std::string, std::vector<int64_t>> name2regst_desc_id_;
  MsgHandler msg_handler_;
  ActorContext* actor_ctx_;
  HashSet<int64_t> eord_regst_desc_ids_;
  int64_t remaining_eord_cnt_;

  HashMap<int64_t, std::vector<std::unique_ptr<Regst>>> produced_regsts_;
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
  std::vector<ActorMsg> sync_msg_queue_;
  bool is_kernel_launch_synchronized_;
  std::vector<int64_t> tmp_regst_desc_id_vec_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_LAZY_ACTOR_ACTOR_H_
