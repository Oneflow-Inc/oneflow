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
#ifndef ONEFLOW_CORE_PROFILER_PROFILER_H_
#define ONEFLOW_CORE_PROFILER_PROFILER_H_

#include "oneflow/core/common/util.h"
#include "json.hpp"

namespace oneflow {

using json = nlohmann::json;

namespace profiler {

void ParseBoolFlagFromEnv(const std::string& env_var, bool* flag);

void NameThisHostThread(const std::string& name);

void RangePush(const std::string& name);

void RangePop();

void LogHostMemoryUsage(const std::string& name);

class RangeGuardCtx;

class RangeGuard final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RangeGuard);
  explicit RangeGuard(const std::string& name);
  ~RangeGuard();

 private:
  std::shared_ptr<RangeGuardCtx> ctx_;
};

class HostMemoryGuard final {
  using JSONCallback = std::function<void(json&)>;

 public:
  OF_DISALLOW_COPY_AND_MOVE(HostMemoryGuard);
  explicit HostMemoryGuard(const std::string& name);
  explicit HostMemoryGuard(const std::string& name, JSONCallback json_callback);
  ~HostMemoryGuard();

 private:
  std::string name_;
  int64_t start_vm_size_;
  int64_t start_rss_size_;
  JSONCallback json_callback_;
};

#ifdef OF_ENABLE_PROFILER
#define OF_PROFILER_NAME_THIS_HOST_THREAD(name) ::oneflow::profiler::NameThisHostThread(name)
#define OF_PROFILER_ONLY_CODE(...) __VA_ARGS__
#define OF_PROFILER_RANGE_PUSH(name) ::oneflow::profiler::RangePush(name)
#define OF_PROFILER_RANGE_POP() ::oneflow::profiler::RangePop()
#define OF_PROFILER_RANGE_GUARD(name) \
  ::oneflow::profiler::RangeGuard OF_PP_CAT(_of_profiler_range_guard_, __COUNTER__)(name)
#define OF_PROFILER_LOG_HOST_MEMORY_USAGE(name) ::oneflow::profiler::LogHostMemoryUsage(name)
#define OF_PROFILER_LOG_HOST_MEMORY_GUARD(name)                                       \
  ::oneflow::profiler::HostMemoryGuard OF_PP_CAT(_of_profiler_log_host_memory_guard_, \
                                                 __COUNTER__)(name)
#define OF_PROFILER_LOG_HOST_MEMORY_GUARD_WITH_JSON(name, json)                       \
  ::oneflow::profiler::HostMemoryGuard OF_PP_CAT(_of_profiler_log_host_memory_guard_, \
                                                 __COUNTER__)(name, json)
#else
#define OF_PROFILER_ONLY_CODE(...)
#define OF_PROFILER_RANGE_PUSH(name)
#define OF_PROFILER_RANGE_POP()
#define OF_PROFILER_RANGE_GUARD(name)
#define OF_PROFILER_NAME_THIS_HOST_THREAD(name)
#define OF_PROFILER_LOG_HOST_MEMORY_USAGE(name)
#define OF_PROFILER_LOG_HOST_MEMORY_GUARD(name)
#define OF_PROFILER_LOG_HOST_MEMORY_GUARD_WITH_JSON(name, json)
#endif

}  // namespace profiler

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PROFILER_PROFILER_H_
