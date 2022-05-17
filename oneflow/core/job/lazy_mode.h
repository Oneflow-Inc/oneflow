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
#ifndef ONEFLOW_CORE_JOB_LAZY_MODE_H_
#define ONEFLOW_CORE_JOB_LAZY_MODE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class LazyMode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyMode);
  LazyMode() = delete;
  ~LazyMode() = delete;

  static bool is_enabled();
  class Guard {
   public:
    explicit Guard(bool enabled) : prev_mode_(LazyMode::is_enabled()) {
      LazyMode::set_enabled(enabled);
    }
    ~Guard() { LazyMode::set_enabled(prev_mode_); }

   private:
    bool prev_mode_;
  };

 private:
  static bool* get_mode_ptr();
  static void set_enabled(bool enabled);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_LAZY_MODE_H_
