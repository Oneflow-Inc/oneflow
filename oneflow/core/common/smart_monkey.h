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
#ifndef ONEFLOW_CORE_COMMON_SMART_MONKEY_H_
#define ONEFLOW_CORE_COMMON_SMART_MONKEY_H_

#include <string>
#include <set>
#include <vector>
#include "oneflow/core/common/chaos.h"

namespace oneflow {
namespace chaos {

class SourceCodePositionScope final {
 public:
  explicit SourceCodePositionScope(const std::string* src_code_pos);
  ~SourceCodePositionScope();
};

class SmartMonkey : public Monkey {
 public:
  SmartMonkey() = default;
  ~SmartMonkey() override {}

  bool Fail() override;

 private:
  std::set<std::vector<const std::string*>> stacks_;
};

#ifdef ENABLE_CHAOS

#define OF_SMART_MONKEY_SOURCE_CODE_POS_SCOPE()                                                    \
  ::oneflow::chaos::SourceCodePositionScope OF_CHAOS_CAT(src_code_pos_scope_, __COUNTER__)(([]() { \
    static std::string pos(std::string(__FILE__ ":") + std::to_string(__LINE__));                  \
    return &pos;                                                                                   \
  })())

#else  // ENABLE_CHAOS

#define OF_SMART_MONKEY_SOURCE_CODE_POS_SCOPE()

#endif  // ENABLE_CHAOS

}  // namespace chaos
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SMART_MONKEY_H_
