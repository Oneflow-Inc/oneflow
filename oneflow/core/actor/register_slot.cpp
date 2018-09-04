#include "oneflow/core/actor/register_slot.h"

namespace oneflow {

bool RegstSlot::FindTheRegstDescId(int64_t regst_desc_id) const {
  CHECK(is_inited_);
  return regst_desc_id2regsts_.find(regst_desc_id) != regst_desc_id2regsts_.end();
}

const std::deque<Regst*>& RegstSlot::RegstDeq4RegstDescId(int64_t regst_desc_id) const {
  CHECK(is_inited_);
  return regst_desc_id2regsts_.at(regst_desc_id);
}

int RegstSlot::PushBackRegst(Regst* regst) {
  CHECK(is_inited_);
  auto it = regst_desc_id2regsts_.find(regst->regst_desc_id());
  if (it == regst_desc_id2regsts_.end()) { return -1; }
  if (it->second.empty()) { available_regst_desc_cnt_ += 1; }
  it->second.push_back(regst);
  return 0;
}

int RegstSlot::PopFrontRegst(int64_t regst_desc_id) {
  CHECK(is_inited_);
  auto it = regst_desc_id2regsts_.find(regst_desc_id);
  if (it == regst_desc_id2regsts_.end()) { return -1; }
  CHECK(it->second.empty() == false);
  it->second.pop_front();
  if (it->second.empty()) { available_regst_desc_cnt_ -= 1; }
  return 0;
}

void RegstSlot::InsertRegstDescId(int64_t regst_desc_id) {
  CHECK(is_inited_ == false);
  CHECK(regst_desc_id2regsts_.emplace(regst_desc_id, std::deque<Regst*>()).second);
}

Regst* RegstSlot::Front(int64_t regst_desc_id) const {
  CHECK(is_inited_);
  auto it = regst_desc_id2regsts_.find(regst_desc_id);
  if (it == regst_desc_id2regsts_.end()) { return nullptr; }
  if (it->second.empty()) { return nullptr; }
  return it->second.front();
}

Regst* RegstSlot::SoleFront() const {
  CHECK(is_inited_);
  CHECK_EQ(1, total_regst_desc_cnt());
  auto it = regst_desc_id2regsts_.begin();
  if (it->second.empty()) { return nullptr; }
  return it->second.front();
}

Regst* RegstSlot::FirstFront() const {
  CHECK(is_inited_);
  CHECK_GE(total_regst_desc_cnt(), 1);
  auto it = regst_desc_id2regsts_.begin();
  if (it->second.empty()) { return nullptr; }
  return it->second.front();
}

void RegstSlot::InitedDone() {
  CHECK(is_inited_ == false);
  is_inited_ = true;
}

void RegstSlot::ForEachCurFrontRegst(std::function<void(const Regst*)> handler) const {
  for (const auto& kv : regst_desc_id2regsts_) {
    if (kv.second.empty() == false) { handler(kv.second.front()); }
  }
}

void RegstSlot::ForEachCurRegstDeq(std::function<void(const std::deque<Regst*>&)> handler) const {
  for (const auto& kv : regst_desc_id2regsts_) { handler(kv.second); }
}

}  // namespace oneflow
