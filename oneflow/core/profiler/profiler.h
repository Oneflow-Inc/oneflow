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

namespace oneflow {

namespace profiler {

void NameThisHostThread(const std::string& name);

void RangePush(const std::string& name);

void RangePop();

void LogHostMemoryUsage(const std::string& name);

void ProfilerStart();

void ProfilerStop();

class RangeGuardCtx;

class RangeGuard final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RangeGuard);
  explicit RangeGuard(const std::string& name);
  ~RangeGuard();

 private:
  std::shared_ptr<RangeGuardCtx> ctx_;
};

#define OF_PROFILER_NAME_THIS_HOST_THREAD(name) ::oneflow::profiler::NameThisHostThread(name)

#ifdef OF_ENABLE_PROFILER
#define OF_PROFILER_ONLY_CODE(...) __VA_ARGS__
#define OF_PROFILER_RANGE_PUSH(name) ::oneflow::profiler::RangePush(name)
#define OF_PROFILER_RANGE_POP() ::oneflow::profiler::RangePop()
#define OF_PROFILER_RANGE_GUARD(name) \
  ::oneflow::profiler::RangeGuard OF_PP_CAT(_of_profiler_range_guard_, __COUNTER__)(name)
#define OF_PROFILER_LOG_HOST_MEMORY_USAGE(name) ::oneflow::profiler::LogHostMemoryUsage(name)
#else
#define OF_PROFILER_ONLY_CODE(...)
#define OF_PROFILER_RANGE_PUSH(name)
#define OF_PROFILER_RANGE_POP()
#define OF_PROFILER_RANGE_GUARD(name)
#define OF_PROFILER_LOG_HOST_MEMORY_USAGE(name)
#endif

void EnableProfiler(bool use_cpu, bool use_cuda, bool record_shapes, bool record_attrs,
                    bool record_bandwidth);

// DisableProfilerAndReturnResult will return a json of profile results.
Maybe<std::string> DisableProfilerAndReturnResult();

Maybe<std::string> StartRecord(const std::string& name);

Maybe<void> EndRecord(const std::string& event_recorder_key);

}  // namespace profiler

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PROFILER_PROFILER_H_
