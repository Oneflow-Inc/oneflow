#include "oneflow/core/actor/register_slot.h"

namespace oneflow {

bool RegstSlot::FindTheRegstDescId(int64_t regst_desc_id) const {
  return regst_desc_id2regsts_.find(regst_desc_id) != regst_desc_id2regsts_.end();
}

void RegstSlot::PushBackRegst(int64_t regst_desc_id, Regst* regst) {
  CHECK(is_inited_);
  auto& deq = regst_desc_id2regsts_.at(regst_desc_id);
  if (deq.empty()) { available_regst_desc_cnt_ += 1; }
  deq.push_back(regst);
}

void RegstSlot::PopFrontRegst(int64_t regst_desc_id) {
  CHECK(is_inited_);
  auto& deq = regst_desc_id2regsts_.at(regst_desc_id);
  CHECK(deq.empty() == false);
  deq.pop_front();
  if (deq.empty()) { available_regst_desc_cnt_ -= 1; }
}

void RegstSlot::InsertOrPushBackRegst(int64_t regst_desc_id, Regst* regst) {
  regst_desc_id2regsts_[regst_desc_id].push_back(regst);
}

Regst* RegstSlot::Front(int64_t regst_desc_id) const {
  CHECK(is_inited_);
  auto it = regst_desc_id2regsts_.find(regst_desc_id);
  if (it == regst_desc_id2regsts_.end()) { return nullptr; }
  if (it->second.empty()) { return nullptr; }
  return it->second.front();
}

Regst* RegstSlot::SoleFront(int64_t regst_desc_id) const {
  CHECK(is_inited_);
  CHECK_EQ(1, total_regst_desc_cnt());
  auto it = regst_desc_id2regsts_.begin();
  if (it->second.empty()) { return nullptr; }
  return it->second.front();
}

void RegstSlot::LockCurSlot() {
  CHECK(is_inited_ == false);
  is_inited_ = true;
  available_regst_desc_cnt_ = total_regst_desc_cnt();
}

}  // namespace oneflow
