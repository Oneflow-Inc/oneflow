#ifndef ONEFLOW_CORE_ACTOR_OP_ACTOR_CONTEXT_H_
#define ONEFLOW_CORE_ACTOR_OP_ACTOR_CONTEXT_H_

#include "oneflow/core/actor/regst_handler.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

namespace actor {

using MsgHandler = std::function<int(const ActorMsg&)>;

class OpActorCtx {
 public:
  explicit OpActorCtx(MsgHandler, const std::vector<RegstHandlerIf*>&,
                      const std::shared_ptr<void>&);
  virtual ~OpActorCtx() = default;

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

  MsgHandler initial_msg_handler() const;

 private:
  struct ExecKernel {
    std::unique_ptr<const Kernel> kernel;
    HashMap<std::string, int64_t> bn_in_op2regst_desc_id;
  };
  void InitDeviceCtx(const ThreadCtx&);
  void InitRegstHandlers(const TaskProto&);

  int64_t actor_id_;
  int64_t act_id_;
  std::unique_ptr<ParallelContext> parallel_ctx_;
  std::unique_ptr<DeviceCtx> device_ctx_;
  std::unique_ptr<NewKernelCtx> kernel_ctx_;
  MsgHandler initial_msg_handler_;
  HashMap<std::string, std::vector<int64_t>> name2regst_desc_id_;
  std::vector<ExecKernel> exec_kernel_vec_;
  HashMap<int64_t, std::vector<std::unique_ptr<Regst>>> produced_regsts_;

  HashMap<std::string, std::unique_ptr<RegstHandlerIf>> handlers_;
  HashMap<int64_t, RegstHandlerIf*> regst_desc_id2handler_;
};

OpActorCtx* CreateOpActorCtx(const TaskType&);

}  // namespace actor

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_OP_ACTOR_CONTEXT_H_
