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
#ifndef ONEFLOW_CORE_PROFILER_KINETO_SHIM_H_
#define ONEFLOW_CORE_PROFILER_KINETO_SHIM_H_

#if defined(WITH_CUDA)

#include <string>
#include <memory>
#include <set>

namespace libkineto {

enum class ActivityType;
class ActivityTraceInterface;

}  // namespace libkineto

namespace oneflow {

namespace profiler {

enum class ActivityType {
  CPU = 0,
  CUDA,
};

using interface_trace_t = libkineto::ActivityTraceInterface;

struct ActivityTraceWrapper {
  explicit ActivityTraceWrapper(std::unique_ptr<interface_trace_t> trace);
  ActivityTraceWrapper() = default;
  ActivityTraceWrapper(ActivityTraceWrapper&&) = default;
  ActivityTraceWrapper(const ActivityTraceWrapper&) = delete;
  explicit operator bool() const;
  void save(const std::string& path);

  const std::unique_ptr<interface_trace_t>& get() { return trace_; }

 private:
  std::unique_ptr<interface_trace_t> trace_;
  bool saved_ = false;  // Kineto's save is destructive
};

using ActivitySet = std::set<ActivityType>;
void PrepareTrace(const bool cpuOnly, const ActivitySet& activities);
void StartTrace();
ActivityTraceWrapper StopTrace();

}  // namespace profiler
}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_PROFILER_KINETO_SHIM_H_
