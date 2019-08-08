#ifndef ONEFLOW_CORE_ACTOR_OP_ACTOR_CONTEXT_H_
#define ONEFLOW_CORE_ACTOR_OP_ACTOR_CONTEXT_H_

namespace oneflow {

namespace actor {

class MsgHandler;

class OpActorCtx {
 public:
  using ProducedRegstType = HashMap<int64_t, std::vector<std::unique_ptr<Regst>>>;

  OpActorCtx(MsgHandler, const std::vector<RegstHandlerIf*>&);
  virtual ~OpActorCtx() = default;

  int64_t act_id() const { return act_id_; }
  std::unique_ptr<DeviceCtx>& mut_device_ctx() { return device_ctx_; }
  const ParallelContext* parallel_ctx() const { return parallel_ctx_.get(); }

  void Init(const TaskProto&, const ThreadCtx&);
  void UpdateWithRegstMsg(const ActorMsg&);
  void UpdateWithEordMsg(const ActorMsg&);
  void UpdateWithCmdMsg(const ActorMsg&);

  bool IsReady4Act() const;
  void Act();
  void HandleRegstMsgAfterAct();
  bool NoLongerConsumeRegst() const;

  void ProcessMsgFromConsumers();
  void RecvAllProducedMsg();

  MsgHandler initial_msg_handler() const;

  void SetInitMsgHandler(MsgHanler handler);
  void InsertRegstPattern(RegstHandlerIfIf*);

 private:
  int64_t actor_id_;
  int64_t act_id_;
  std::unique_ptr<ParallelContext> parallel_ctx_;
  std::unique_ptr<DeviceCtx> device_ctx_;
  MsgHandler initial_msg_handler_;
  HashMap<std::string, std::vector<int64_t>> name2regst_desc_id_;
  std::vector<ExecKernel> exec_kernel_vec_;
  ProducedRegstType produced_regsts_;

  HashMap<std::string, std::unique_ptr<RegstHandlerIfIf>> handlers_;
  HashMap<int64_t, RegstHandlerIfIf*> regst_desc_id2handler_;
};

}

}

#endif // ONEFLOW_CORE_ACTOR_OP_ACTOR_CONTEXT_H_
