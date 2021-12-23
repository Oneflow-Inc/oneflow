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
#include "oneflow/core/common/chaos.h"

namespace oneflow {
namespace chaos {

namespace {

bool* MutThreadLocalEnableChaos() {
  thread_local bool enable_chaos = true;
  return &enable_chaos;
}

}  // namespace

bool ThreadLocalEnableChaos() { return *MutThreadLocalEnableChaos(); }

ChaosModeScope::ChaosModeScope(bool enable_mode) {
  old_enable_mode_ = *MutThreadLocalEnableChaos();
  *MutThreadLocalEnableChaos() = enable_mode;
}

ChaosModeScope::~ChaosModeScope() { *MutThreadLocalEnableChaos() = old_enable_mode_; }

namespace {

class NiceMonkey : public Monkey {
 public:
  NiceMonkey() = default;
  ~NiceMonkey() override {}

  bool Fail() override { return false; }
};

}  // namespace

namespace {

std::unique_ptr<Monkey>* MutThreadLocalMonkey() {
  thread_local std::unique_ptr<Monkey> monkey_ = std::make_unique<NiceMonkey>();
  return &monkey_;
}

}  // namespace

Monkey* ThreadLocalMonkey() { return MutThreadLocalMonkey()->get(); }

MonkeyScope::MonkeyScope(std::unique_ptr<Monkey>&& monkey) {
  old_monkey_ = std::move(*MutThreadLocalMonkey());
  *MutThreadLocalMonkey() = std::move(monkey);
}

MonkeyScope::~MonkeyScope() { *MutThreadLocalMonkey() = std::move(old_monkey_); }

}  // namespace chaos
}  // namespace oneflow
