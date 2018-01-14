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

namespace oneflow {

class Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Actor);
  virtual ~Actor() = default;

  void Init(const TaskProto&, const ThreadCtx&);

  // 1: success, and actor finish
  // 0: success, and actor not finish
  int ProcessMsg(const ActorMsg& msg) { return (this->*msg_handler_)(msg); }

  int64_t machine_id() const;
  int64_t thrd_id() const;
  int64_t actor_id() const { return actor_id_; }

 protected:
  struct ExecKernel {
    std::unique_ptr<const Kernel> kernel;
    HashMap<std::string, int64_t> bn_in_op2regst_desc_id;
  };
  using MsgHandler = int (Actor::*)(const ActorMsg&);

  // Util
  Actor() = default;
  int64_t GetReservedWorkStreamId(int64_t reserved_id);
  int64_t NewWorkStreamId();
  int64_t GetWorkStreamId() const { return device_ctx_->work_stream_id(); }
  const ParallelContext* parallel_ctx() const { return parallel_ctx_.get(); }
  DeviceType GetDeviceType() const;
  virtual void VirtualActorInit(const TaskProto&) {}
  int64_t RegstDescId4Name(const std::string& name) const;
  virtual void InitDeviceCtx(const ThreadCtx&);
  std::unique_ptr<DeviceCtx>& mut_device_ctx() { return device_ctx_; }
  KernelCtx GenDefaultKernelCtx() const;
  const std::vector<ExecKernel>& exec_kernel_vec() { return exec_kernel_vec_; }
  virtual void ForEachCurReadableRegst(std::function<void(const Regst*)>) {}
  virtual void SetReadableRegstInfo(const Regst*, ReadableRegstInfo*);

  // Msg Handler
  void set_msg_handler(MsgHandler val) { msg_handler_ = val; }
#define OF_SET_MSG_HANDLER(val)                                   \
  do {                                                            \
    LOG(INFO) << "actor " << actor_id() << " switch to " << #val; \
    set_msg_handler(static_cast<MsgHandler>(val));                \
  } while (0)

  // Common Handlers
  virtual int HandlerNormal(const ActorMsg& msg) = 0;
  int HandlerZombie(const ActorMsg& msg);

  // Act
  void ActUntilFail();
  virtual void Act() = 0;
  virtual bool IsReadReady() = 0;
  virtual bool IsReadAlwaysUnReadyFromNow() = 0;
  virtual bool IsWriteReady();
  virtual void AsyncReturnAllReadableRegst() = 0;
  void DecreaseRemainingEordCnt();
  int TrySwitchToZombieOrFinish();

  // Async Do on device_ctx_
  void AsyncLaunchKernel(const KernelCtx&,
                         std::function<Regst*(int64_t)> Regst4RegstDescId);
  void AsyncLaunchRecurrentKernel(
      const KernelCtx&,
      std::function<Regst*(int64_t, const std::string&)> FindRegst);
  void AsyncSendRegstMsgToConsumer(std::function<bool(Regst*)> RegstPreProcess,
                                   std::function<bool(int64_t)> IsAllowedActor);
  void AsyncSendRegstMsgToConsumer(std::function<bool(Regst*)> RegstPreProcess);
  void AsyncSendRegstMsgToConsumer(std::function<bool(int64_t)> IsAllowedActor);
  void AsyncSendRegstMsgToConsumer();
  void AsyncSendEORDMsgToConsumers(int64_t regst_desc_id);
  void AsyncSendEORDMsgForAllProducedRegstDesc();
  void AsyncSendRegstMsgToProducer(Regst*);
  void AsyncSendRegstMsgToProducer(Regst*, int64_t producer);
  void AsyncDo(std::function<void()>);

  // Status of Produced Registers
  int TryUpdtStateAsProducedRegst(Regst* regst);
  Regst* GetCurWriteableRegst(int64_t regst_desc_id);
  Regst* GetCurWriteableRegst(const std::string& name);
  Regst* GetCurSoleWriteableRegst();
  int64_t total_reading_cnt() const { return total_reading_cnt_; }

 private:
  int64_t actor_id_;
  int64_t act_id_;
  std::unique_ptr<ParallelContext> parallel_ctx_;
  std::vector<ExecKernel> exec_kernel_vec_;
  HashMap<int64_t, std::vector<std::unique_ptr<Regst>>> produced_regsts_;
  HashMap<std::string, int64_t> name2regst_desc_id_;
  MsgHandler msg_handler_;
  std::unique_ptr<DeviceCtx> device_ctx_;
  CudaStreamHandle cuda_handle_;

  // Status of Produced Registers
  HashMap<int64_t, std::deque<Regst*>> writeable_produced_regst_;
  HashMap<Regst*, int64_t> produced_regst2reading_cnt_;
  int64_t writeable_produced_regst_desc_num_;
  int64_t total_reading_cnt_;
  int64_t remaining_eord_cnt_;
};

void AddActorCreator(TaskType task_type, std::function<Actor*()> creator);
std::unique_ptr<Actor> NewActor(const TaskProto&, const ThreadCtx&);

template<TaskType task_type, typename T>
struct ActorRegistry {
  ActorRegistry() {
    AddActorCreator(task_type, []() { return new T; });
  }
};

#define REGISTER_ACTOR(TaskType, ActorType)            \
  static ActorRegistry<TaskType, ActorType> OF_PP_CAT( \
      g_actor_##ActorType##registry_var, __LINE__)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACTOR_H_
