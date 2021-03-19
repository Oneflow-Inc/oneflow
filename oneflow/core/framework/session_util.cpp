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
#include <mutex>
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/session_util.h"

namespace oneflow {

namespace {

std::mutex* GlobalSessionUtilMutex() {
  static std::mutex global_id2session_map_mutex;
  return &global_id2session_map_mutex;
}

HashMap<int64_t, std::shared_ptr<Session>>* GlobalId2SessionMap() {
  static HashMap<int64_t, std::shared_ptr<Session>> id2session_map;
  return &id2session_map;
}

int64_t* DefaultSessionId() {
  static int64_t default_sess_id;
  return &default_sess_id;
}

Maybe<void> SetDefaultSessionId(int64_t val) {
  int64_t* id = DefaultSessionId();
  *id = val;
  return Maybe<void>::Ok();
}

}  // namespace

Session::Session(int64_t id, const std::shared_ptr<vm::cfg::InstructionListProto>& instruction_list,
                 const std::shared_ptr<eager::cfg::EagerSymbolList>& symbol_list)
    : id_(id),
      instruction_list_(instruction_list),
      eager_symbol_list_(symbol_list),
      is_mirrored_strategy_enabled_stack_(new std::vector<bool>()) {}

int64_t Session::id() const { return id_; }

std::shared_ptr<vm::cfg::InstructionListProto> Session::instruction_list() const {
  return instruction_list_;
}

std::shared_ptr<eager::cfg::EagerSymbolList> Session::eager_symbol_list() const {
  return eager_symbol_list_;
}

Maybe<void> Session::PushMirroredStrategyEnabled(bool is_mirrored) {
  is_mirrored_strategy_enabled_stack_->push_back(is_mirrored);
  return Maybe<void>::Ok();
}
Maybe<void> Session::PopMirroredStrategyEnabled() {
  is_mirrored_strategy_enabled_stack_->pop_back();
  return Maybe<void>::Ok();
}

Maybe<bool> Session::IsMirroredStrategyEnabled() const {
  // Mirrored strategy is enabled by default.
  return is_mirrored_strategy_enabled_stack_->size() == 0
         || is_mirrored_strategy_enabled_stack_->back();
}
Maybe<bool> Session::IsConsistentStrategyEnabled() const {
  bool is_mirrored_enabled = JUST(IsMirroredStrategyEnabled());
  return !is_mirrored_enabled;
}

Maybe<int64_t> GetDefaultSessionId() { return *(DefaultSessionId()); }

Maybe<Session> RegsiterSession(int64_t id) {
  std::shared_ptr<Session> sess =
      std::make_shared<Session>(id, std::make_shared<vm::cfg::InstructionListProto>(),
                                std::make_shared<eager::cfg::EagerSymbolList>());
  std::unique_lock<std::mutex> lock(*GlobalSessionUtilMutex());
  auto* id2session_map = GlobalId2SessionMap();
  CHECK_OR_RETURN(id2session_map->find(id) == id2session_map->end());
  (*id2session_map)[id] = sess;
  JUST(SetDefaultSessionId(id));
  return id2session_map->at(id);
}

Maybe<Session> GetDefaultSession() {
  int64_t default_sess_id = JUST(GetDefaultSessionId());
  std::unique_lock<std::mutex> lock(*GlobalSessionUtilMutex());
  auto* id2session_map = GlobalId2SessionMap();
  CHECK_OR_RETURN(id2session_map->find(default_sess_id) != id2session_map->end());
  return id2session_map->at(default_sess_id);
}

Maybe<void> ClearSessionById(int64_t id) {
  std::unique_lock<std::mutex> lock(*GlobalSessionUtilMutex());
  auto* id2session_map = GlobalId2SessionMap();
  CHECK_OR_RETURN(id2session_map->find(id) != id2session_map->end());
  id2session_map->erase(id);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
