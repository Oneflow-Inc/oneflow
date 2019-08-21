#ifndef ONEFLOW_CORE_ACTOR_OP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_OP_ACTOR_H_

#include "oneflow/core/actor/regst_handler.h"
#include "oneflow/core/actor/new_actor.h"

namespace oneflow {

namespace actor {

class OpActor : public NewActor {
 public:
  static int HandlerNormal(OpActor*, const ActorMsg&);
  static int HandlerZombie(OpActor*, const ActorMsg&);

  int64_t actor_id() const { return actor_id_; }
  int64_t act_id() const { return act_id_; }
  std::unique_ptr<DeviceCtx>& mut_device_ctx() { return device_ctx_; }
  const ParallelContext* parallel_ctx() const { return parallel_ctx_.get(); }
  DeviceType GetDeviceType() const;

  void Init(const TaskProto&, const ThreadCtx&);
  void UpdateWithRegstMsg(const ActorMsg&);
  void UpdateWithProducedRegstMsg(const ActorMsg&);
  void UpdateWithEordMsg(const ActorMsg&);
  void UpdateWithCmdMsg(const ActorMsg&);

  bool IsReady() const;
  void Act();
  void HandleRegstMsgAfterAct();
  bool NoLongerConsumeRegst() const;
  bool NoLongerConsumedByOthers() const;
  void SendEordMsgForProducedRegst() const;

  MsgHandler initial_msg_handler() const;

 protected:
  OpActor() = default;
  virtual ~OpActor() = default;

  void set_other_val(std::shared_ptr<void> other_val) { kernel_ctx_->other = other_val; }

 private:
  struct ExecKernel {
    std::unique_ptr<const Kernel> kernel;
    HashMap<std::string, int64_t> bn_in_op2regst_desc_id;
  };
  void InitDeviceCtx(const ThreadCtx&);
  void InitRegstHandlersFromProto(const TaskProto&);

  virtual void InitMsgHandler() = 0;
  virtual void InitOtherVal() = 0;
  virtual void SetOtherVal4CurAct(void*) = 0;

  int64_t actor_id_;
  int64_t act_id_;
  std::unique_ptr<ParallelContext> parallel_ctx_;
  std::unique_ptr<DeviceCtx> device_ctx_;
  std::unique_ptr<NewKernelCtx> kernel_ctx_;
  HashMap<std::string, std::vector<int64_t>> name2regst_desc_id_;
  std::vector<ExecKernel> exec_kernel_vec_;
  HashMap<int64_t, std::vector<std::unique_ptr<Regst>>> produced_regsts_;

  HashMap<std::string, std::unique_ptr<RegstHandlerIf>> handlers_;
  HashMap<int64_t, RegstHandlerIf*> regst_desc_id2handler_;
};

#define OF_SET_OP_ACTOR_MSG_HANDLER(actor, handler)                           \
  do {                                                                        \
    LOG(INFO) << "actor " << actor->actor_id() << " switch to " << #handler;  \
    actor->set_msg_handler(std::bind(handler, actor, std::placeholders::_1)); \
  } while (0)
#define OF_CLEAR_OP_ACTOR_MSG_HANDLER(actor)                                  \
  do {                                                                        \
    LOG(INFO) << "actor " << actor->actor_id() << " release its msg handler"; \
    actor->set_msg_handler(MsgHandler());                                     \
  } while (0)

}  // namespace actor

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_OP_ACTOR_H_
