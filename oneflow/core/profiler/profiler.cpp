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
#ifdef OF_ENABLE_PROFILER
#include <nvtx3/nvToolsExt.h>
#include <sys/syscall.h>
#include <iostream>
#include <cuda_profiler_api.h>
#include "oneflow/core/device/cuda_util.h"
#endif  // OF_ENABLE_PROFILER

namespace oneflow {

namespace profiler {

void NameThisHostThread(const std::string& name) {
#ifdef OF_ENABLE_PROFILER
  static thread_local std::unique_ptr<std::string> thread_name_prefix;
  if (!thread_name_prefix) {
    thread_name_prefix.reset(
        new std::string(GetStringFromEnv("ONEFLOW_PROFILER_HOST_THREAD_NAME_PREFIX", "")));
  }
  const std::string name_with_prefix = *thread_name_prefix + name;
  nvtxNameOsThreadA(syscall(SYS_gettid), name_with_prefix.c_str());
#endif  // OF_ENABLE_PROFILER
}

void RangePush(const std::string& name) {
#ifdef OF_ENABLE_PROFILER
  nvtxRangePushA(name.c_str());
#endif  // OF_ENABLE_PROFILER
}

RangeScope::RangeScope(const std::string& name) { RangePush(name); }

void RangePop() {
#ifdef OF_ENABLE_PROFILER
  nvtxRangePop();
#endif  // OF_ENABLE_PROFILER
}

RangeScope::~RangeScope() { RangePop(); }

#ifdef OF_ENABLE_PROFILER

class RangeGuardCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RangeGuardCtx);
  explicit RangeGuardCtx(nvtxRangeId_t range_id) : range_id_(range_id) {}
  ~RangeGuardCtx() = default;

  nvtxRangeId_t range_id() const { return range_id_; }

 private:
  nvtxRangeId_t range_id_;
};
#else
class RangeGuardCtx {};
#endif  // OF_ENABLE_PROFILER

RangeGuard::RangeGuard(const std::string& name) {
#ifdef OF_ENABLE_PROFILER
  nvtxRangeId_t range_id = nvtxRangeStartA(name.c_str());
  ctx_.reset(new RangeGuardCtx(range_id));
#endif  // OF_ENABLE_PROFILER
}

RangeGuard::~RangeGuard() {
#ifdef OF_ENABLE_PROFILER
  nvtxRangeEnd(ctx_->range_id());
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

}  // namespace profiler

}  // namespace oneflow
