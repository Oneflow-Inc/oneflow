#ifndef ONEFLOW_CORE_ACTOR_OP_ACTOR_CONTEXT_H_
#define ONEFLOW_CORE_ACTOR_OP_ACTOR_CONTEXT_H_

namespace oneflow {

namespace actor {

enum class RegstPattern {
  kCtrl = 0,
  kNaive,
  kInplace,
  kCustomized
};

class RegstPatternWrapper {
 public:
  RegstPatternWrapper() = default;
  virtual ~RegstPatternWrapper() = default;

  void Init(const TaskProto& task_proto) {
    DerivedInit();
    consumed_rs_.InitDone();
    produced_rs_.InitDone();
  }
  const RegstSlot& consumed_rs() { return consumed_rs_; }
  const RegstSlot& produced_rs()  { return produced_rs_; }
  virtual RegstPattern type() = 0;

 protected:
  RegstPatternWrapper() = default;
  ~RegstPatternWrapper() = default;

  void InsertNewRegstDescId(bool is_produced, int64_t regst_desc_id) {
    if (is_produced) {
      produced_rs_.InsertRegstDescId(regst_desc_id);
      regst_desc_id2produced_.emplace(regst_desc_id, true);
    } else {
      consumed_rs_.InsertRegstDescId(regst_desc_id);
      regst_desc_id2produced_.emplace(regst_desc_id, false);
    }
  }

 private:
  virtual void DerivedInit(const TaskProto&) = 0;

  RegstSlot consumed_rs_;
  RegstSlot produced_rs_;
  HashMap<int64_t, bool> regst_desc_id2produced_; // true for produced, false for consumed
};

class CtrlPatternWrapper final : public RegstPatternWrapper {
 public:
  RegstPattern type() override { return RegstPattern::kCtrl; }
  void DerivedInit(const TaskProto&) override;
};

class NaivePatternWrapper final : public RegstPatternWrapper {
 public:
  RegstPattern type() override { return RegstPattern::kNaive; }
  void DerivedInit(const TaskProto&) override;
};

class InplacePatternWrapper final : public RegstPatternWrapper {
 public:
  RegstPattern type() override { return RegstPattern::kInplace; }
  void DerivedInit(const TaskProto&) override;
};

class OpActorCtx {
 public:
  void Init(const TaskProto&, const ThreadCtx&);
  void UpdateWithCmdMsg(const ActorMsg&);
  void UpdateWithEordMsg(const ActorMsg&);
  void UpdateWithRegstMsg(const ActorMsg&);

  bool IsReady4Act() const;
  void Act();
  bool EndOfRead() const;

  void ProcessMsgFromConsumers();
  void RecvAllProducedMsg();

  MsgHandler initial_msg_handler() const { return initial_msg_handler_; }

 protected:

 private:
  int64_t actor_id_;
  int64_t act_id_;
  std::unique_ptr<ParallelContext> parallel_ctx_;
  std::unique_ptr<DeviceCtx> device_ctx_;
  MsgHandler initial_msg_handler_;
  HashMap<std::string, std::vector<int64_t>> name2regst_desc_id_;

  std::vector<ExecKernel> exec_kernel_vec_;
  HashMap<RegstPattern, std::unique_ptr<RegstSlotWrapper>> rs_wrappers_;

  HashSet<int64_t> eord_regst_desc_ids_;
  int64_t remaining_eord_cnt_;
};

class NaiveOpActorCtx final : public OpActorCtx {
 public:
};

}

}

#endif // ONEFLOW_CORE_ACTOR_OP_ACTOR_CONTEXT_H_
