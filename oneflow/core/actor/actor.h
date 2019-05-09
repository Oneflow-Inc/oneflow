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
#include "oneflow/core/persistence/snapshot_manager.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/thread/thread_context.h"
#include "oneflow/core/actor/register_slot.h"

namespace oneflow {

enum class ColIdOrder { kUnCertain = 0, kAscending, kDescending };

bool IsFirstRegstInPieceWithOrder(const Regst*, ColIdOrder);
bool IsLastRegstInPieceWithOrder(const Regst*, ColIdOrder);
bool NeedModelSave(int64_t model_version_id);

class Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Actor);
  virtual ~Actor() = default;

  void Init(const TaskProto&, const ThreadCtx&);

  // 1: success, and actor finish
  // 0: success, and actor not finish
  int ProcessMsg(const ActorMsg& msg) { return (this->*msg_handler_)(msg); }

  int64_t machine_id() const { return Global<IDMgr>::Get()->MachineId4ActorId(actor_id_); }
  int64_t thrd_id() const { return Global<IDMgr>::Get()->ThrdId4ActorId(actor_id_); }
  int64_t actor_id() const { return actor_id_; }

 protected:
  struct ExecKernel {
    std::unique_ptr<const Kernel> kernel;
    HashMap<std::string, int64_t> bn_in_op2regst_desc_id;
  };
  using MsgHandler = int (Actor::*)(const ActorMsg&);
  enum class RegstNameType { kNaive = 0, kCustomized };

  // Util
  Actor() = default;
  const ParallelContext* parallel_ctx() const { return parallel_ctx_.get(); }
  bool ReceiveAllEordMsg() const { return remaining_eord_cnt_ == 0; }
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
  void AsyncSendMsg(const ActorMsg&);
  void HandleProducedNaiveDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess,
                                              std::function<bool(int64_t)> IsAllowedActor);
  void HandleProducedNaiveDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess);
  void HandleProducedNaiveDataRegstToConsumer(std::function<bool(int64_t)> IsAllowedActor);
  void HandleProducedNaiveDataRegstToConsumer();
  void AsyncSendRegstMsgToConsumer(Regst* regst);
  void AsyncSendRegstMsgToConsumer(Regst* regst, std::function<bool(int64_t)> IsAllowedActor);

  void HandleConsumedNaiveDataRegstToProducer(std::function<bool(Regst*)> IsAllowedRegst);
  void AsyncSendRegstMsgToProducer(Regst*);
  void AsyncSendRegstMsgToProducer(Regst*, int64_t producer);
  void AsyncSendEORDMsgForAllProducedRegstDesc();

  // Get Regst
  Regst* GetNaiveCurReadable(int64_t desc_id) { return naive_consumed_rs_.Front(desc_id); }
  Regst* GetNaiveCurReadable(const std::string& name) {
    return GetNaiveCurReadable(Name2SoleRegstDescId(name));
  }
  Regst* GetNaiveCurWriteable(int64_t desc_id) { return naive_produced_rs_.Front(desc_id); }
  Regst* GetNaiveCurWriteable(const std::string& name) {
    return GetNaiveCurWriteable(Name2SoleRegstDescId(name));
  }
  Regst* GetSoleProducedRegst4RegstDescId(int64_t regst_desc_id);

 private:
  int64_t GetGlobalWorkStreamId() const;
  int64_t GetLocalWorkStreamId() const;
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
  bool IsReadReady();
  bool IsWriteReady();

  // NaiveOrCustomized
  void TakeOverNaiveConsumed(const PbMap<std::string, RegstDescIdSet>& consumed_ids);
  void TakeOverNaiveProduced(const PbMap<std::string, RegstDescProto>& produced_ids);

  // Send Msgs
  void AsyncSendNaiveProducedRegstMsgToConsumer();
  virtual void VirtualAsyncSendNaiveProducedRegstMsgToConsumer();
  void AsyncSendInplaceProducedRegstMsgToConsumer();
  void AsyncSendNaiveConsumedRegstMsgToProducer();
  virtual void VirtualAsyncSendNaiveConsumedRegstMsgToProducer();
  void AsyncSendConsumedCtrlRegstMsgToProducer();
  void AsyncSendProducedCtrlRegstMsgToConsumer();
  int64_t HandleRegstToConsumer(Regst* regst, std::function<bool(int64_t)> IsAllowedActor);

  // Customized Consumed virtual func
  virtual void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)>) const {}
  virtual void NormalProcessCustomizedEordMsg(const ActorMsg&) {}
  virtual void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) { UNIMPLEMENTED(); }
  virtual bool IsCustomizedReadReady() { return true; }
  virtual bool IsCustomizedReadAlwaysUnReadyFromNow() { return false; }
  virtual std::pair<RegstNameType, HashSet<std::string>>
  GetNaiveOrCustomizedConsumedRegstDescName() {
    return std::make_pair(RegstNameType::kCustomized, HashSet<std::string>{});
  }
  virtual void AsyncSendCustomizedProducedRegstMsgToConsumer() {}
  virtual void AsyncReturnAllCustomizedReadableRegst() {}

  // Customized Produced virtual func
  virtual void UpdtStateAsCustomizedProducedRegst(Regst* regst) { UNIMPLEMENTED(); }
  virtual bool IsCustomizedWriteReady() { return true; }
  virtual std::pair<RegstNameType, HashSet<std::string>>
  GetNaiveOrCustomizedProducedRegstDescName() {
    return std::make_pair(RegstNameType::kCustomized, HashSet<std::string>{});
  }
  virtual void AsyncSendCustomizedConsumedRegstMsgToProducer() {}

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
  HashMap<int64_t, int64_t> inplace_out2in_;
};

class ScopedActEventRecorder;

std::unique_ptr<Actor> NewActor(const TaskProto&, const ThreadCtx&);

#define REGISTER_ACTOR(task_type, ActorType) REGISTER_CLASS(task_type, Actor, ActorType)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACTOR_H_
