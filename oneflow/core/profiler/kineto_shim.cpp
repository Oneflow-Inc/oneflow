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

#if defined(WITH_CUDA)

#include "oneflow/core/profiler/kineto_shim.h"
#include "libkineto.h"

namespace oneflow {

namespace profiler {
namespace {

const std::set<libkineto::ActivityType> cpuTypes{
    libkineto::ActivityType::CPU_OP,          libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::USER_ANNOTATION, libkineto::ActivityType::EXTERNAL_CORRELATION,
    libkineto::ActivityType::CUDA_RUNTIME,  // something like cudaLaunchKernel
    libkineto::ActivityType::PYTHON_FUNCTION,
};

const std::set<libkineto::ActivityType> cudaTypes = {
    libkineto::ActivityType::GPU_MEMCPY, libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::CONCURRENT_KERNEL,  // cuda kernel
    // CUDA_RUNTIME appears in both cpuTypes and cudaTypes.
    libkineto::ActivityType::CUDA_RUNTIME,  // something like cudaLaunchKernel
};
}  // namespace

ActivityTraceWrapper::ActivityTraceWrapper(std::unique_ptr<interface_trace_t> trace)
    : trace_(std::move(trace)), saved_{false} {}

ActivityTraceWrapper::operator bool() const { return trace_ != nullptr; }

void ActivityTraceWrapper::save(const std::string& path) {
  //   TORCH_CHECK(!saved_, "Trace is already saved.");
  //   TORCH_CHECK(trace_ != nullptr, "Missing trace.")
  trace_->save(path);
  saved_ = true;
}

void PrepareTrace(const bool cpuOnly, const ActivitySet& activities) {
  if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init(/*cpuOnly=*/cpuOnly, /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }

  if (!libkineto::api().isProfilerInitialized()) { libkineto::api().initProfilerIfRegistered(); }

  std::set<libkineto::ActivityType> k_activities;
  if (activities.count(ActivityType::CPU)) {
    k_activities.insert(cpuTypes.begin(), cpuTypes.end());
  }
  if (activities.count(ActivityType::CUDA)) {
    k_activities.insert(cudaTypes.begin(), cudaTypes.end());
  }

  libkineto::api().activityProfiler().prepareTrace(k_activities);
}

void StartTrace() { libkineto::api().activityProfiler().startTrace(); }

ActivityTraceWrapper StopTrace() {
  return ActivityTraceWrapper{libkineto::api().activityProfiler().stopTrace()};
}

}  // namespace profiler
}  // namespace oneflow

#endif  // WITH_CUDA
