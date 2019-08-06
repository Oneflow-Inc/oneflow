#ifndef ONEFLOW_CORE_ACTOR_REGST_PATTERN_WRAPPER_H_
#define ONEFLOW_CORE_ACTOR_REGST_PATTERN_WRAPPER_H_

namespace oneflow {

namespace actor {

struct ActorMsgUtil {
  static void AsyncSendMsg(OpActorCtx* op_actor_ctx, const ActorMsg& msg) {
    std::function<void()> callback = [msg]() { Global<ActorMsgBus>::Get()->SendMsg(msg); };
    if (Global<IDMgr>::Get()->GlobalWorkStreamId4ActorId(op_actor_ctx->actor_id());
        == Global<IDMgr>::Get()->GlobalWorkStreamId4ActorId(msg.dst_actor_id())) {
      callback();
    } else {
      op_actor_ctx->mut_device_ctx()->AddCallBack(callback);
    }
  }
};

class RegstPatternWrapperIf {
 public:
  virtual void Init(const TaskProto&, const ProducedRegstType&, OpActorCtx*) = 0;
  virtual std::string type() = 0;

  virtual Regst* GetRegstByRegstDescId() const = 0;

  virtual void UpdateWithEordMsg(const ActorMsg&) = 0;
  virtual void UpdateWithRegstMsg(const ActorMsg&) = 0;
  virtual void UpdateWithProducedRegstMsg(const ActorMsg&) = 0;

  virtual bool IsReady4Act() const = 0;
  virtual void HandleRegstMsgAfterAct() = 0;
  virtual bool NoLongerConsumeRegst() const = 0;
};

class NormalPatternWrapper : public RegstPatternWrapperIf {
 public:
  void Init(const TaskProto&, const ProducedRegstType&, OpActorCtx*) override final;
  bool NoLongerConsumeRegst() const override final { return (eord_cnt_ == consumed_rs_.total_reading_cnt()); }
  void UpdateWithRegstMsg(const ActorMsg&) override final;
  void UpdateWithEordMsg(const ActorMsg&) override final;
  bool IsReady4Act() const override final;
  Regst* GetRegstByRegstDescId() const override final;
  void HandleRegstMsgAfterAct() override final;

  void ForEachRegstDescId(std::function<void(int64_t)>) const;

 protected:
  NormalPatternWrapper() = default;
  ~NormalPatternWrapper() = default;

  OpActorCtx* op_actor_ctx() { return op_actor_ctx_; }
  RegstSlot* mut_consumed_rs() { return &consumed_rs_; }
  RegstSlot* mut_produced_rs()  { return &produced_rs_; }

  int64_t ReadingCnt4ProducedRegst(Regst* regst) const {
    return produced_regst2reading_cnt_.at(regst);
  }
  void UpdateReadingCnt4ProducedRegst(Regst* regst, int64_t update_val) {
    produced_regst2reading_cnt_.at(regst) += update_val;
    total_reading_cnt_ += udpate_val;
  }

  void InsertNewRegstDescId(bool is_produced, int64_t regst_desc_id) {
    if (is_produced) {
      produced_rs_.InsertRegstDescId(regst_desc_id);
    } else {
      consumed_rs_.InsertRegstDescId(regst_desc_id);
      consumed_regst2eord_.emplace(regst_desc_id, false);
    }
  }

 private:
  virtual void DerivedInit(const TaskProto&) = 0;
  virtual void UpdateWithConsumedRegstMsg(Regst*) = 0;
  virtual void HandleConsumedRegstAfterAct() = 0;
  virtual void HandleProducedRegstAfterAct() = 0;

  OpActorCtx* op_actor_ctx_;

  RegstSlot consumed_rs_;
  RegstSlot produced_rs_;

  HashSet<int64_t, bool> consumed_regst2eord_;
  int64_t eord_cnt_;
  HashMap<Regst*, int64_t> produced_regst2reading_cnt_;
  int64_t total_reading_cnt_;
  // HashMap<int64_t, int64_t> produced_regst2expected_act_id_;
};

class CtrlPatternWrapper final : public NormalPatternWrapper {
 public:
  std::string type() override { return "Ctrl"; }
  void DerivedInit(const TaskProto&) override;
  void UpdateWithConsumedRegstMsg(Regst*) override;
  void UpdateWithProducedRegstMsg(Regst*) override;
  void HandleConsumedRegstAfterAct() override;
  void HandleProducedRegstAfterAct() override;
};

class NaivePatternWrapper final : public NormalPatternWrapper {
 public:
  std::string type() override { return "Naive"; }
  void DerivedInit(const TaskProto&) override;
  void UpdateWithConsumedRegstMsg(Regst*) override;
  void UpdateWithProducedRegstMsg(Regst*) override;
  void HandleConsumedRegstAfterAct() override;
  void HandleProducedRegstAfterAct() override;
};

class InplacePatternWrapper final : public NormalPatternWrapper {
 public:
  std::string type() override { return "Inplace"; }
  void DerivedInit(const TaskProto&) override;
  void UpdateWithConsumedRegstMsg(Regst*) override;
  void UpdateWithProducedRegstMsg(Regst*) override;
  void HandleConsumedRegstAfterAct() override;
  void HandleProducedRegstAfterAct() override;

 private:
  HashMap<int64_t, int64_t> inplace_pair_in2out_;
  HashMap<int64_t, int64_t> inplace_pair_out2in_;
};

}

}

#endif // ONEFLOW_CORE_ACTOR_REGST_PATTERN_WRAPPER_H_
