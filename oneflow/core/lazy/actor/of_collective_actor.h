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
#ifndef ONEFLOW_CORE_LAZY_ACTOR_OF_COLLECTIVE_ACTOR_H_
#define ONEFLOW_CORE_LAZY_ACTOR_OF_COLLECTIVE_ACTOR_H_

#include "oneflow/core/lazy/actor/actor_base.h"
#include "oneflow/core/lazy/actor/actor_message_bus.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/lazy/actor/register_slot.h"
#include "oneflow/core/lazy/actor/of_collective_boxing_actor_context.h"

namespace oneflow {

#define MYLOG VLOG(100)

class OfCollectiveActor final: public ActorBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveActor);
  OfCollectiveActor() = default;
  ~OfCollectiveActor() override = default;

  void Init(const JobDesc* job_desc, ActorContext* actor_ctx) override;

  // 1: success, and actor finish
  // 0: success, and actor not finish
  int ProcessMsg(const ActorMsg& msg) override { return (this->*msg_handler_)(msg); }

  int64_t machine_id() const { return MachineId4ActorId(actor_id_); }
  int64_t actor_id() const { return actor_id_; }
  int64_t job_id() const { return job_id_; }

  int64_t GetUpstreamId() { return nego_tree_info_.upstream_id; }
  std::vector<int64_t> GetDownstreamId() { return nego_tree_info_.downstream_id; }

 private:
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
  enum class RegstNameType { kNaive = 0, kCustomized };

  // Msg Handler
  using MsgHandler = int (OfCollectiveActor::*)(const ActorMsg&);
  void set_msg_handler(MsgHandler val) { msg_handler_ = val; }
#define OF_SET_MSG_HANDLER(val)                                 \
  do {                                                          \
    VLOG(3) << "actor " << actor_id() << " switch to " << #val; \
    set_msg_handler(static_cast<MsgHandler>(val));              \
  } while (0)

  int HandlerNormal(const ActorMsg& msg);
  int HandlerZombie(const ActorMsg& msg);

  void NormalProcessNaiveReadableDataRegstMsg(const std::deque<Regst*>&) {}
  bool NormalTryProcessReadableMsgFromOtherMachine(const ActorMsg&) { return false; }
  int TryUpdtStateAsProducedRegst(Regst* regst);

  // ready
  bool IsReadReady() const;
  bool IsWriteReady() const;

  // Act
  void Act();
  void AsyncLaunchKernel(std::function<Regst*(int64_t)> Regst4RegstDescId);

  // Send Msg
  void EnqueueAsyncMsg(const ActorMsg&);
  void AsyncSendRegstMsgToProducer(Regst*);
  void AsyncSendRegstMsgToProducer(Regst*, int64_t producer);
  void AsyncSendQueuedMsg();
  void AddCallback(std::function<void()> callback);
  void AsyncSendEORDMsgForAllProducedRegstDesc();
  // transmit regsts after act
  void AsyncSendNaiveProducedRegstMsgToConsumer();
  void HandleProducedNaiveDataRegstToConsumer();
  int64_t HandleRegstToConsumer(Regst* regst);
  void AsyncSendProducedCtrlRegstMsgToConsumer();
  bool IsProducedCtrlRegstDescId(int64_t regst_desc_id) {
    return produced_ctrl_regst_desc_ids_.find(regst_desc_id) != produced_ctrl_regst_desc_ids_.end();
  }
  bool ProducedCtrlRegstValid(int64_t regst_desc_id) const { return true; }
  void HandleProducedInplaceDataRegstToConsumer();
  void AsyncSendNaiveConsumedRegstMsgToProducer();
  void HandleConsumedNaiveDataRegstToProducer();
  void AsyncSendConsumedCtrlRegstMsgToProducer();
  bool IsConsumedCtrlRegstDescId(int64_t regst_desc_id) {
    return consumed_ctrl_regst_desc_ids_.find(regst_desc_id) != consumed_ctrl_regst_desc_ids_.end();
  }
  bool ConsumedCtrlRegstValid(int64_t regst_desc_id) const { return true; }
  void AsyncRetInplaceConsumedRegstIfNoConsumer();

  // Util
  void IncreaseTotalReadingCnt(int64_t val) { total_reading_cnt_ += val; }
  int64_t ReadingCnt4ProducedRegst(Regst* regst) const;
  void IncreaseReadingCnt4ProducedRegst(Regst* regst, int64_t val);
  bool ReceiveAllEordMsg() const { return remaining_eord_cnt_ == 0; }


  // manage registers
  virtual void TakeOverInplaceConsumedAndProduced(
      const PbMap<std::string, RegstDescProto>& produced_ids);
  void TakeOverNaiveConsumed(const PbMap<std::string, RegstDescIdSet>& consumed_ids);
  void TakeOverNaiveProduced(const PbMap<std::string, RegstDescProto>& produced_ids);
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName() {
    return std::make_pair(RegstNameType::kCustomized, HashSet<std::string>{});
  }
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedProducedRegstDescName() {
    return std::make_pair(RegstNameType::kCustomized, HashSet<std::string>{});
  }
  void InitBnInOp2BlobInfo(const TaskProto& task_proto);

  // Collective Negotiation
  enum class CollectiveStatus {
    kInvalid = 0,
    kLocalReady,
    kDownstreamReady,
    kCanAct
  };
  bool IsDownstreamReady() const;
  bool HasUpstream() const { return nego_tree_info_.upstream_id != -1; }
  bool HasDownstream() const { return nego_tree_info_.downstream_id.size() > 0; }
  bool CanAct() const { return collective_status_ == CollectiveStatus::kCanAct; }
  void ResetCollectiveStatus();
  void ReactToNegoCmd(const ActorMsg& msg);
  void SyncSendMsg(const ActorMsg&);
  HashMap<CollectiveNegoCmd, std::string> print_nego_cmd_;
  HashMap<CollectiveStatus, std::string> print_status_;
  HashMap<ActorMsgType, std::string> print_actor_msg_type_;
  boxing::of_collective::RuntimeNegoTreeInfo nego_tree_info_;
  int64_t received_downstream_ready_cnt_;
  HashSet<int64_t> ready_downstream_id_;
  CollectiveStatus collective_status_;
  // In case get kCollectiveReady when local not ready.
  HashMap<int64_t, ActorMsg> cached_nego_ready_msg_;
  bool is_nego_root_;
  bool is_nego_leaf_;

  ActorContext* actor_ctx_;  
  int64_t actor_id_;
  int64_t thrd_id_;
  int64_t job_id_;
  std::string op_name_;

  MsgHandler msg_handler_;

  ExecKernel ek_;
  bool is_kernel_launch_synchronized_;

  // regsts
  HashMap<int64_t, std::vector<std::unique_ptr<Regst>>> produced_regsts_;
  HashMap<std::string, std::vector<int64_t>> name2regst_desc_id_;
  HashMap<Regst*, int64_t> produced_regst2reading_cnt_;
  // eord
  int64_t remaining_eord_cnt_;
  HashSet<int64_t> eord_regst_desc_ids_;
  // inplace
  HashMap<int64_t, int64_t> inplace_regst_desc_id_in2out_;
  HashMap<int64_t, int64_t> inplace_regst_desc_id_out2in_;
  RegstSlot inplace_consumed_rs_;
  RegstSlot inplace_produced_rs_;
  bool is_inplace_consumed_eord_;
  HashSet<int64_t> inplace_in_ids_with_no_out_consumed_;
  // naive
  RegstSlot naive_produced_rs_;
  RegstSlot naive_consumed_rs_;
  bool is_naive_consumed_eord_;
  // ctrl
  HashSet<int64_t> produced_ctrl_regst_desc_ids_;
  HashSet<int64_t> consumed_ctrl_regst_desc_ids_;

  std::vector<int64_t> tmp_regst_desc_id_vec_;

  int64_t total_reading_cnt_;

  std::deque<ActorMsg> async_msg_queue_;

};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_LAZY_ACTOR_OF_COLLECTIVE_ACTOR_H_
