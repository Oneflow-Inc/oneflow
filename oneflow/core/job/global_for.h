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
#ifndef ONEFLOW_CORE_JOB_GLOBAL_FOR_H_
#define ONEFLOW_CORE_JOB_GLOBAL_FOR_H_

#include <utility>

#include "oneflow/core/common/global.h"

namespace oneflow {

class ForSession {};
class ForEnv {};

class EagerExecution {};

struct DTRConfig {
  bool is_enabled;
  float memory_threshold;
  int debug_level;
  std::string heuristic;
  DTRConfig(bool is_enabled, float memory_threshold, int debug_level, const std::string &heuristic)
      : is_enabled(is_enabled),
        memory_threshold(memory_threshold),
        debug_level(debug_level),
        heuristic(heuristic) {}
};

class MultiClient {};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_GLOBAL_FOR_H_
