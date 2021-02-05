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
#include "oneflow/core/framework/session_util.h"

namespace oneflow {

Session::Session(const std::shared_ptr<vm::cfg::InstructionListProto>& instruction_list,
                 const std::shared_ptr<eager::cfg::EagerSymbolList>& symbol_list)
    : instruction_list_(instruction_list), eager_symbol_list_(symbol_list) {}
std::shared_ptr<vm::cfg::InstructionListProto> Session::instruction_list() const {
  return instruction_list_;
}
std::shared_ptr<eager::cfg::EagerSymbolList> Session::eager_symbol_list() const {
  return eager_symbol_list_;
}

Maybe<Session> GetDefaultSession() {
  static std::shared_ptr<Session> default_sess;
  if (!default_sess) {
    default_sess = std::make_shared<Session>(std::make_shared<vm::cfg::InstructionListProto>(),
                                             std::make_shared<eager::cfg::EagerSymbolList>());
  }
  return default_sess;
}
Maybe<void> ResetDefaultSession() {
  std::shared_ptr<Session> default_sess = JUST(GetDefaultSession());
  CHECK_OR_RETURN(default_sess);
  default_sess.reset();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
