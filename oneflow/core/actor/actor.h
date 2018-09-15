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

  // Util
  Actor() = default;
  const ParallelContext* parallel_ctx() const { return parallel_ctx_.get(); }
  bool ReceiveAllEordMsg() const { return remaining_eord_cnt_ == 0; }
  DeviceType GetDeviceType() const;
  virtual void VirtualActorInit(const TaskProto&) {}
  int64_t Name2SoleRegstDescId(const std::string& name) const;
  const std::vector<int64_t>& Name2RegstDescId(const std::string& name) const;
  virtual void InitDeviceCtx(const ThreadCtx&);
  std::unique_ptr<DeviceCtx>& mut_device_ctx() { return device_ctx_; }
  KernelCtx GenDefaultKernelCtx() const;
  const std::vector<ExecKernel>& exec_kernel_vec() { return exec_kernel_vec_; }
  virtual void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)>) const {}
  virtual void SetReadableRegstInfo(const Regst*, ReadableRegstInfo*) const;
  void ForEachCurNaiveReadableRegst(std::function<void(const Regst*)>) const;
  void ForEachCurConsumedCtrlRegst(std::function<void(const Regst*)>) const;

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

  virtual void NormalProcessCustomizedEordMsg(const ActorMsg&) {}
  virtual void NormalProcessNaiveReadableRegstMsg(const std::deque<Regst*>&) {}
  virtual void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) { UNIMPLEMENTED(); }
  virtual bool NormalTryProcessReadableMsgFromOtherMachine(const ActorMsg&) { return false; }

  // Act
  void ActUntilFail();
  virtual void Act(std::function<bool(Regst*)>* IsNaiveAllowedReturnToProducer) { Act(); }
  virtual void Act() { UNIMPLEMENTED(); }
  virtual bool IsCustomizedReadReady() { return true; }
  virtual bool IsCustomizedReadAlwaysUnReadyFromNow() { return false; }
  bool IsWriteReady();
  virtual void AsyncReturnAllCustomizedReadableRegst() {}
  virtual int64_t ActNumForEachOutput(int64_t regst_desc_id) const { return 1; }
  virtual bool CheckOutputActId(int64_t regst_desc_id) const {
    return true;  // TODO(jiyuan): figure out the ActNumForEachOutput of the model regsts to MdSave
                  // area
  }

  virtual bool ConsumedCtrlRegstValid(int64_t regst_desc_id) const { return true; }
  virtual bool ProducedCtrlRegstValid(int64_t regst_desc_id) const { return true; }

  // Async Do on device_ctx_
  void AsyncLaunchKernel(const KernelCtx&, std::function<Regst*(int64_t)> Regst4RegstDescId);
  void AsyncLaunchKernel(const KernelCtx&);
  void AsyncSendRegstMsgToConsumer(std::function<bool(Regst*)> RegstPreProcess,
                                   std::function<bool(int64_t)> IsAllowedActor);
  void AsyncSendRegstMsgToConsumer(std::function<bool(Regst*)> RegstPreProcess);
  void AsyncSendRegstMsgToConsumer(std::function<bool(int64_t)> IsAllowedActor);
  void AsyncSendRegstMsgToConsumer();
  void AsyncSendEORDMsgToConsumers(int64_t regst_desc_id);
  void AsyncSendEORDMsgForAllProducedRegstDesc();
  void AsyncSendRegstMsgToProducer(Regst*);
  void AsyncSendRegstMsgToProducer(Regst*, int64_t producer);
  void AsyncDo(std::function<void()> func) { device_ctx_->AddCallBack(func); }

  // Status of Produced Registers
  Regst* GetCurWriteableRegst(int64_t regst_desc_id);
  Regst* GetCurWriteableRegst(const std::string& name);
  Regst* GetCurSoleWriteableRegst();

  // Status Of Naive Consumed Registers
  virtual std::pair<bool, std::vector<std::string>> GetNaiveConsumedRegstDescName() {
    return {false, {}};
  }
  Regst* GetNaiveCurReadable(int64_t desc_id) { return naive_consumed_data_rs_.Front(desc_id); }
  Regst* GetNaiveSoleCurReadable() { return naive_consumed_data_rs_.SoleFront(); }
  void ForEachNaiveCurReadable(std::function<void(const Regst*)>) const;
  Regst* GetNaiveFirstCurReadable() { return naive_consumed_data_rs_.FirstFront(); }
  Regst* GetSoleProducedRegst(int64_t regst_desc_id);
  int64_t GetSoleProducedDataRegstDescId() const;

  void DecreaseActualWriteableProducedDataRegstDescNum(int64_t amount) {
    actual_writeable_produced_data_regst_desc_num_ -= amount;
  }

 private:
  bool IsReadReady();
  bool IsCtrlReady();
  int ProcessWriteableCtrlRegstMsg(const ActorMsg& msg);
  int ProcessReadableCtrlRegstMsg(const ActorMsg& msg);
  void AsyncSendEORDMsgForAllProducedCtrlRegstDesc();
  void AsyncSendCtrlRegstMsgToProducer();
  void AsyncSendCtrlRegstMsgToConsumer();
  int TryUpdtStateAsProducedRegst(Regst* regst);
  void TakeOverNaiveConsumed(const PbMap<std::string, RegstDescIdSet>& consumed_ids);
  void AddNaiveConsumed(const RegstDescIdSet&);
  void AsyncSendMsg(const ActorMsg&);
  int64_t GetGlobalWorkStreamId() const;
  int64_t GetLocalWorkStreamId() const;
  virtual bool NeedCollectActEvent() const {
    return Global<RuntimeCtx>::Get()->NeedCollectActEvent();
  }
  void TryLogActEvent(const std::function<void()>& Callback) const;

  int64_t actor_id_;
  int64_t act_id_;
  std::unique_ptr<ParallelContext> parallel_ctx_;
  std::vector<ExecKernel> exec_kernel_vec_;
  HashMap<std::string, std::vector<int64_t>> name2regst_desc_id_;
  MsgHandler msg_handler_;
  std::unique_ptr<DeviceCtx> device_ctx_;
  HashSet<int64_t> eord_regst_desc_ids_;
  std::unique_ptr<CudaStreamHandle> cuda_handle_;
  int64_t remaining_eord_cnt_;

  // Status of Produced Registers
  HashMap<int64_t, std::vector<std::unique_ptr<Regst>>> produced_data_regsts_;
  HashMap<int64_t, int64_t> produced_data_regst2expected_act_id_;
  HashMap<Regst*, int64_t> produced_data_regst2reading_cnt_;
  int64_t total_reading_data_cnt_;

  RegstSlot writeable_produced_data_rs_;
  int64_t actual_writeable_produced_data_regst_desc_num_;
  // Status of Naive Consumed Registers
  RegstSlot naive_consumed_data_rs_;
  bool is_naive_consumed_data_eord_;

  // Status of Control Registers
  HashMap<int64_t, std::vector<std::unique_ptr<Regst>>> produced_ctrl_regst_;
  HashMap<int64_t, int64_t> produced_ctrl_regst2expected_act_id_;
  HashMap<Regst*, int64_t> produced_ctrl_regst2reading_cnt_;
  int64_t total_reading_ctrl_cnt_;

  RegstSlot writeable_produced_ctrl_rs_;
  RegstSlot consumed_ctrl_rs_;
  bool is_consumed_ctrl_eord_;
};

class ScopedActEventRecorder;

std::unique_ptr<Actor> NewActor(const TaskProto&, const ThreadCtx&);

#define REGISTER_ACTOR(task_type, ActorType) REGISTER_CLASS(task_type, Actor, ActorType)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACTOR_H_
