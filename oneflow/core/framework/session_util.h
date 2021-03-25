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
#ifndef ONEFLOW_CORE_FRAMEWORK_SESSION_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_SESSION_UTIL_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/vm/instruction.cfg.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/eager/eager_symbol.cfg.h"

namespace oneflow {

class Session {
 public:
  Session(int64_t id);
  Session(const Session&) = delete;
  Session(Session&&) = delete;
  ~Session() = default;

  int64_t id() const;
  const std::shared_ptr<vm::InstructionMsgList>& instruction_list() const;
  std::shared_ptr<eager::cfg::EagerSymbolList> eager_symbol_list() const;

  std::shared_ptr<const std::vector<bool>> is_mirrored_strategy_enabled_stack() const {
    return is_mirrored_strategy_enabled_stack_;
  }
  Maybe<void> PushMirroredStrategyEnabled(bool is_mirrored);
  Maybe<void> PopMirroredStrategyEnabled();
  Maybe<bool> IsMirroredStrategyEnabled() const;
  Maybe<bool> IsConsistentStrategyEnabled() const;

 private:
  int64_t id_;
  std::shared_ptr<vm::InstructionMsgList> instruction_list_;
  std::shared_ptr<eager::cfg::EagerSymbolList> eager_symbol_list_;
  std::shared_ptr<std::vector<bool>> is_mirrored_strategy_enabled_stack_;
};

Maybe<int64_t> GetDefaultSessionId();
Maybe<Session> RegsiterSession(int64_t id);
Maybe<Session> GetDefaultSession();
Maybe<void> ClearSessionById(int64_t id);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SESSION_UTIL_H_
