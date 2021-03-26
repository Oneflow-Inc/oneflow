/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/actor/register_slot.h"

namespace oneflow {

bool RegstSlot::HasRegstDescId(int64_t regst_desc_id) const {
  CHECK(is_inited_);
  return regst_desc_id2regsts_.find(regst_desc_id) != regst_desc_id2regsts_.end();
}

const std::deque<Regst*>& RegstSlot::RegstDeq4RegstDescId(int64_t regst_desc_id) const {
  CHECK(is_inited_);
  return regst_desc_id2regsts_.at(regst_desc_id);
}

int RegstSlot::TryPushBackRegst(Regst* regst) {
  return TryPushBackRegst(regst, regst->regst_desc_id());
}

int RegstSlot::TryPushBackRegst(Regst* regst, int64_t regst_desc_id) {
  CHECK(is_inited_);
  auto it = regst_desc_id2regsts_.find(regst_desc_id);
  if (it == regst_desc_id2regsts_.end()) { return -1; }
  if (it->second.empty()) { available_regst_desc_cnt_ += 1; }
  it->second.push_back(regst);
  return 0;
}

int RegstSlot::TryPopFrontRegst(int64_t regst_desc_id) {
  CHECK(is_inited_);
  auto it = regst_desc_id2regsts_.find(regst_desc_id);
  if (it == regst_desc_id2regsts_.end()) { return -1; }
  CHECK(it->second.empty() == false);
  it->second.pop_front();
  if (it->second.empty()) { available_regst_desc_cnt_ -= 1; }
  return 0;
}

void RegstSlot::PopFrontRegsts(const std::vector<int64_t>& regst_desc_ids) {
  CHECK(is_inited_);
  for (int64_t regst_desc_id : regst_desc_ids) { CHECK_EQ(0, TryPopFrontRegst(regst_desc_id)); }
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

void RegstSlot::ForChosenFrontRegst(std::function<bool(int64_t)> IsChosenRegstDescId,
                                    std::function<void(Regst*)> Handler) const {
  for (const auto& kv : regst_desc_id2regsts_) {
    if (IsChosenRegstDescId(kv.first)) {
      CHECK(kv.second.empty() == false);
      Handler(kv.second.front());
    }
  }
}

void RegstSlot::ForChosenFrontRegst(
    std::function<bool(int64_t)> IsChosenRegstDescId,
    std::function<void(int64_t regst_desc_id, Regst*)> Handler) const {
  for (const auto& kv : regst_desc_id2regsts_) {
    if (IsChosenRegstDescId(kv.first)) {
      CHECK(kv.second.empty() == false);
      Handler(kv.first, kv.second.front());
    }
  }
}

void RegstSlot::ForChosenRegstDeq(std::function<bool(int64_t)> IsChosenRegstDescId,
                                  std::function<void(const std::deque<Regst*>&)> Handler) const {
  for (const auto& kv : regst_desc_id2regsts_) {
    if (IsChosenRegstDescId(kv.first)) { Handler(kv.second); }
  }
}

void RegstSlot::ForChosenRegstDeq(
    std::function<bool(int64_t)> IsChosenRegstDescId,
    std::function<void(int64_t regst_desc_id, const std::deque<Regst*>&)> Handler) const {
  for (const auto& kv : regst_desc_id2regsts_) {
    if (IsChosenRegstDescId(kv.first)) { Handler(kv.first, kv.second); }
  }
}

void RegstSlot::ForEachFrontRegst(std::function<void(Regst*)> Handler) const {
  ForChosenFrontRegst([](int64_t) { return true; }, Handler);
}

void RegstSlot::ForEachFrontRegst(
    std::function<void(int64_t regst_desc_id, Regst*)> Handler) const {
  ForChosenFrontRegst([](int64_t) { return true; }, Handler);
}

void RegstSlot::ForEachRegstDeq(std::function<void(const std::deque<Regst*>&)> Handler) const {
  ForChosenRegstDeq([](int64_t) { return true; }, Handler);
}

}  // namespace oneflow
