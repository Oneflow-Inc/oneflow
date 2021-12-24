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
#ifndef ONEFLOW_CORE_COMMON_CHAOS_H_
#define ONEFLOW_CORE_COMMON_CHAOS_H_

#include <memory>

namespace oneflow {
namespace chaos {

bool ThreadLocalEnableChaos();

class ChaosModeScope final {
 public:
  explicit ChaosModeScope(bool enable_mode);
  ~ChaosModeScope();

 private:
  bool old_enable_mode_;
};

class Monkey {
 public:
  virtual ~Monkey() {}

  size_t failed_cnt() const { return failed_cnt_; }

  bool FailIf() {
    bool ret = Fail();
    failed_cnt_ += static_cast<int>(ret);
    return ret;
  }
  virtual bool Fail() = 0;

 protected:
  Monkey() : failed_cnt_() {}

 private:
  size_t failed_cnt_;
};

Monkey* ThreadLocalMonkey();

class MonkeyScope {
 public:
  MonkeyScope(std::unique_ptr<Monkey>&& monkey);
  ~MonkeyScope();

  const Monkey& current_monkey() const { return *current_monkey_; }

 private:
  std::unique_ptr<Monkey> old_monkey_;
  const Monkey* current_monkey_;
};

#ifdef OF_ENABLE_CHAOS

#define OF_CHAOS_MODE_SCOPE(enable_mode) \
  ::oneflow::chaos::ChaosModeScope OF_CHAOS_CAT(chaos_mode_scope_, __COUNTER__)(enable_mode)

#define OF_CHAOS_BOOL_EXPR(expr)                                               \
  ({                                                                           \
    bool expr_ret = static_cast<bool>(expr);                                   \
    if (::oneflow::chaos::ThreadLocalMonkey()->FailIf()) { expr_ret = false; } \
    expr_ret;                                                                  \
  })

#else  // OF_ENABLE_CHAOS

#define OF_CHAOS_MODE_SCOPE(enable_mode)

#define OF_CHAOS_BOOL_EXPR(expr) (expr)

#endif  // OF_ENABLE_CHAOS

#define OF_CHAOS_CAT(a, b) OF_CHAOS_CAT_I(a, b)
#define OF_CHAOS_CAT_I(a, b) a##b
}  // namespace chaos
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CHAOS_H_
