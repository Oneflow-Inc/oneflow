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

#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/profiler/profile_manager.h"
#include "oneflow/core/profiler/kineto_shim.h"
#include "oneflow/core/profiler/event_recorder.h"
#include "oneflow/core/vm/vm_util.h"
#ifdef WITH_CUDA
#include "oneflow/core/device/cuda_util.h"
#include <nvtx3/nvToolsExt.h>
#include <sys/syscall.h>
#include <iostream>
#include <cuda_profiler_api.h>
#endif  // WITH_CUDA

namespace oneflow {

namespace profiler {

void NameThisHostThread(const std::string& name) {
#ifdef WITH_CUDA
  static thread_local std::unique_ptr<std::string> thread_name_prefix;
  if (!thread_name_prefix) {
    thread_name_prefix.reset(
        new std::string(GetStringFromEnv("ONEFLOW_PROFILER_HOST_THREAD_NAME_PREFIX", "")));
  }
  const std::string name_with_prefix = *thread_name_prefix + name;
  nvtxNameOsThreadA(syscall(SYS_gettid), name_with_prefix.c_str());
#endif  // WITH_CUDA
}

void RangePush(const std::string& name) {
#ifdef OF_ENABLE_PROFILER
  nvtxRangePushA(name.c_str());
#endif  // OF_ENABLE_PROFILER
}

void RangePop() {
#ifdef OF_ENABLE_PROFILER
  nvtxRangePop();
#endif  // OF_ENABLE_PROFILER
}

RangeGuard::RangeGuard(const std::string& name) {
#ifdef OF_ENABLE_PROFILER
  RangePush(name);
#endif  // OF_ENABLE_PROFILER
}

RangeGuard::~RangeGuard() {
#ifdef OF_ENABLE_PROFILER
  RangePop();
#endif  // OF_ENABLE_PROFILER
}

void LogHostMemoryUsage(const std::string& name) {
#ifdef OF_ENABLE_PROFILER
  int64_t vm_pages;
  int64_t rss_pages;
  std::ifstream ifs("/proc/self/statm");
  ifs >> vm_pages >> rss_pages;
  ifs.close();
  const int64_t page_size = sysconf(_SC_PAGE_SIZE);
  LOG(INFO) << "HostMemoryUsage: " << name << " VM " << vm_pages * page_size << " RSS "
            << rss_pages * page_size;
#endif  // OF_ENABLE_PROFILER
}

void ProfilerStart() {
#ifdef OF_ENABLE_PROFILER
  OF_CUDA_CHECK(cudaProfilerStart());
#endif  // OF_ENABLE_PROFILER
}

void ProfilerStop() {
#ifdef OF_ENABLE_PROFILER
  OF_CUDA_CHECK(cudaProfilerStop());
#endif  // OF_ENABLE_PROFILER
}

void EnableProfiler(bool use_cpu, bool use_cuda, bool record_shapes, bool record_attrs,
                    bool record_bandwidth) {
  CHECK_JUST(vm::ClusterSync());
  if (Singleton<ProfileManager>::Get() == nullptr) {
    Singleton<ProfileManager>::New(use_cpu, use_cuda, record_shapes, record_attrs,
                                   record_bandwidth);
  }
}

// DisableProfilerAndReturnResult will return a json of profile results.
Maybe<std::string> DisableProfilerAndReturnResult() {
  JUST(vm::ClusterSync());
#if defined(WITH_CUDA)
  OF_CUDA_CHECK(cudaDeviceSynchronize());
#endif  // WITH_CUDA
  auto* pmgr = JUST(SingletonMaybe<ProfileManager>());
  std::string results = pmgr->DumpResultsJson();
  Singleton<ProfileManager>::Delete();
  return results;
}

Maybe<std::string> StartRecord(const std::string& name) {
  auto* pmgr = JUST(SingletonMaybe<ProfileManager>());
  JUST(vm::ClusterSync());
  return pmgr->RegisterEventRecorder(profiler::EventRecorder::CreateCustomEventRecorder(name),
                                     name);
}

Maybe<void> EndRecord(const std::string& event_recorder_key) {
  auto* pmgr = JUST(SingletonMaybe<ProfileManager>());
  JUST(vm::ClusterSync());
  pmgr->UnregisterEventRecorder(event_recorder_key);
  return Maybe<void>::Ok();
}

}  // namespace profiler

}  // namespace oneflow
