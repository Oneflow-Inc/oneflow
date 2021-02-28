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
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/session_util.h"

namespace oneflow {

namespace {

Maybe<HashMap<int64_t, std::shared_ptr<Session>>*> GlobalId2SessionMap() {
  static HashMap<int64_t, std::shared_ptr<Session>> id2session_map;
  return &id2session_map;
}

}  // namespace

Session::Session(int64_t id, const std::shared_ptr<vm::cfg::InstructionListProto>& instruction_list,
                 const std::shared_ptr<eager::cfg::EagerSymbolList>& symbol_list)
    : id_(id), instruction_list_(instruction_list), eager_symbol_list_(symbol_list) {}

int64_t Session::id() const { return id_; }

std::shared_ptr<vm::cfg::InstructionListProto> Session::instruction_list() const {
  return instruction_list_;
}

std::shared_ptr<eager::cfg::EagerSymbolList> Session::eager_symbol_list() const {
  return eager_symbol_list_;
}

Maybe<int64_t*> GetDefaultSessionId() {
  static int64_t default_sess_id;
  return &default_sess_id;
}

Maybe<void> SetDefaultSessionId(int64_t val) {
  int64_t* id = JUST(GetDefaultSessionId());
  *id = val;
  return Maybe<void>::Ok();
}

Maybe<void> RegsiterSession(int64_t id, const std::shared_ptr<Session>& sess) {
  auto* id2session_map = JUST(GlobalId2SessionMap());
  CHECK_OR_RETURN(id2session_map->find(id) == id2session_map->end());
  (*id2session_map)[id] = sess;
  JUST(SetDefaultSessionId(id));
  return Maybe<void>::Ok();
}

Maybe<Session> GetDefaultSession() {
  int64_t default_sess_id = *JUST(GetDefaultSessionId());
  auto* id2session_map = JUST(GlobalId2SessionMap());
  CHECK_OR_RETURN(id2session_map->find(default_sess_id) != id2session_map->end());
  return id2session_map->at(default_sess_id);
}

Maybe<void> ClearDefaultSession() {
  int64_t default_sess_id = *JUST(GetDefaultSessionId());
  auto* id2session_map = JUST(GlobalId2SessionMap());
  CHECK_OR_RETURN(id2session_map->find(default_sess_id) != id2session_map->end());
  id2session_map->erase(default_sess_id);
  return Maybe<void>::Ok();
}

Maybe<void> ClearSessionById(int64_t id) {
  auto* id2session_map = JUST(GlobalId2SessionMap());
  CHECK_OR_RETURN(id2session_map->find(id) != id2session_map->end());
  id2session_map->erase(id);
  return Maybe<void>::Ok();
}

Maybe<void> ClearAllSession() {
  auto* id2session_map = JUST(GlobalId2SessionMap());
  id2session_map->clear();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
