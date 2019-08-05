#ifndef ONEFLOW_CORE_ACTOR_REGST_PATTERN_WRAPPER_H_
#define ONEFLOW_CORE_ACTOR_REGST_PATTERN_WRAPPER_H_

namespace oneflow {

namespace actor {

class RegstPatternWrapperIf {
 public:
  virtual void Init(const TaskProto&) = 0;
  virtual std::string type() = 0;
  virtual bool NoLongerConsumeRegst() = 0;
  virtual void UpdateWithRegstMsg(const ActorMsg&) = 0;
  virtual void UpdateWithEordMsg(const ActorMsg&) = 0;
  virtual bool IsReady4Act() const = 0;
};

class NormalPatternWrapper : public RegstPatternWrapperIf {
 public:
  void Init(const TaskProto& task_proto) override final;
  bool NoLongerConsumeRegst() override final { return (eord_cnt_ == consumed_rs_.total_reading_cnt()); }
  void UpdateWithRegstMsg(const ActorMsg&) override final;
  void UpdateWithEordMsg(const ActorMsg&) override final;
  bool IsReady4Act() const override final;

  void ForEachRegstDescId(std::function<void(int64_t)>) const;

  RegstSlot* mut_consumed_rs() { return &consumed_rs_; }
  RegstSlot* mut_produced_rs()  { return &produced_rs_; }

 protected:
  NormalPatternWrapper() = default;
  ~NormalPatternWrapper() = default;

  void InsertNewRegstDescId(bool is_produced, int64_t regst_desc_id) {
    if (is_produced) {
      produced_rs_.InsertRegstDescId(regst_desc_id);
      produced_regst2reading_cnt_.emplace(regst_desc_id, 0);
    } else {
      consumed_rs_.InsertRegstDescId(regst_desc_id);
      consumed_regst2eord_.emplace(regst_desc_id, false);
    }
  }

 private:
  virtual void DerivedInit(const TaskProto&) = 0;
  virtual void UpdateWithConsumedRegstMsg(Regst*) = 0;
  virtual void UpdateWithProducedRegstMsg(Regst*) = 0;

  RegstSlot consumed_rs_;
  RegstSlot produced_rs_;

  HashSet<int64_t, bool> consumed_regst2eord_;
  size_t eord_cnt_;
  HashMap<int64_t, size_t> produced_regst2reading_cnt_;
  size_t total_reading_cnt_;
  // HashMap<int64_t, int64_t> produced_regst2expected_act_id_;
};

class CtrlPatternWrapper final : public NormalPatternWrapper {
 public:
  std::string type() override { return "Ctrl"; }
  void DerivedInit(const TaskProto&) override;
};

class NaivePatternWrapper final : public NormalPatternWrapper {
 public:
  std::string type() override { return "Naive"; }
  void DerivedInit(const TaskProto&) override;
};

class InplacePatternWrapper final : public NormalPatternWrapper {
 public:
  std::string type() override { return "Inplace"; }
  void DerivedInit(const TaskProto&) override;
};


}

}

#endif // ONEFLOW_CORE_ACTOR_REGST_PATTERN_WRAPPER_H_
