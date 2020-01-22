#ifndef ONEFLOW_CORE_ACTOR_REGISTER_SLOT_H_
#define ONEFLOW_CORE_ACTOR_REGISTER_SLOT_H_

#include "oneflow/core/register/register_manager.h"

namespace oneflow {

class RegstSlot final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstSlot);
  RegstSlot() : regst_desc_id2regsts_(), available_regst_desc_cnt_(0), is_inited_(false) {}
  ~RegstSlot() = default;

  bool is_inited() const { return is_inited_; }
  size_t total_regst_desc_cnt() const { return regst_desc_id2regsts_.size(); }
  size_t available_regst_desc_cnt() const { return available_regst_desc_cnt_; }

  bool IsCurSlotReady() const { return available_regst_desc_cnt() == total_regst_desc_cnt(); }
  bool HasRegstDescId(int64_t regst_desc_id) const;
  const std::deque<Regst*>& RegstDeq4RegstDescId(int64_t regst_desc_id) const;

  template<typename P, typename T>
  void ForChosenRegstDeq(P IsChosenRegstDescId, T Handler) const {
    for (const auto& kv : regst_desc_id2regsts_) {
      if (IsChosenRegstDescId(kv.first)) { Handler(kv.second); }
    }
  }

  template<typename P, typename T>
  void ForChosenFrontRegst(P IsChosenRegstDescId, T Handler) const {
    for (const auto& kv : regst_desc_id2regsts_) {
      if (IsChosenRegstDescId(kv.first)) {
        CHECK(kv.second.empty() == false);
        Handler(kv.second.front());
      }
    }
  }

  template<typename T>
  void ForEachFrontRegst(T Handler) const {
    ForChosenFrontRegst([](int64_t) { return true; }, Handler);
  }

  template<typename T>
  void ForEachRegstDeq(T Handler) const {
    ForChosenRegstDeq([](int64_t) { return true; }, Handler);
  }

  Regst* Front(int64_t regst_desc_id) const;
  Regst* SoleFront() const;
  Regst* FirstFront() const;

  // 0: success, -1: cannot find regst_desc_id
  int TryPushBackRegst(Regst* regst);
  int TryPopFrontRegst(int64_t regst_desc_id);

  void PopFrontRegsts(const std::vector<int64_t>& regst_desc_ids);

  void InitedDone();
  void InsertRegstDescId(int64_t regst_desc_id);

 private:
  std::map<int64_t, std::deque<Regst*>> regst_desc_id2regsts_;
  size_t available_regst_desc_cnt_;
  bool is_inited_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REGISTER_SLOT_H_
