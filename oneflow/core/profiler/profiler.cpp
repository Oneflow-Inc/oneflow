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
#include <chrono>
#include <stack>
#include <map>
#include <unordered_map>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>

#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/profiler/profile_manager.h"
#include "oneflow/core/profiler/kineto_shim.h"
#include "oneflow/core/profiler/event_recorder.h"
#include "oneflow/core/vm/vm_util.h"
#ifdef WITH_CUDA
#include "oneflow/core/device/cuda_util.h"
#endif  // WITH_CUDA
#ifdef OF_ENABLE_PROFILER
#include <nvtx3/nvToolsExt.h>
#include <sys/syscall.h>
#include <iostream>
#include <cuda_profiler_api.h>
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

uint64_t nanos() {
  int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
                   .count();
  return ns;
}

class Profiler {
 public:
  Profiler() = default;
  virtual ~Profiler() {
    std::cout << "========================== PROFILE RESULT =========================="
              << std::endl;
    Print();
  }

  void Push(const std::string& frame) {
    uint64_t start_ns = nanos();
    stack_.push(Frame{frame, start_ns});
  }

  void Pop() {
    uint64_t end_ns = nanos();
    const auto& frame = stack_.top();
    uint64_t time = end_ns - frame.start_ns;
    auto& item = items_[frame.name];
    item.instances += 1;
    item.total_time += time;
    item.elapsed.emplace_back(time);
    if (time < item.minium) { item.minium = time; }
    if (time > item.maxium) { item.maxium = time; }
    stack_.pop();
  }

  void Print() {
    std::map<uint64_t, std::string> sorted_items;
    double sum = 0;
    for (const auto& it : items_) {
      sum += it.second.total_time;
      sorted_items.emplace(it.second.total_time, it.first);
    }
    const char* sep = "  ";

    std::cout.setf(std::ios::fixed);
    std::cout.precision(1);

    std::cout << std::setw(7) << "Time(%)";
    std::cout << sep;
    std::cout << std::setw(15) << "Total Time (ns)";
    std::cout << sep;
    std::cout << std::setw(10) << "Instances";
    std::cout << sep;
    std::cout << std::setw(10) << "Average";
    std::cout << sep;
    std::cout << std::setw(10) << "Minimum";
    std::cout << sep;
    std::cout << std::setw(10) << "Maximum";
    std::cout << sep;
    std::cout << std::left << std::setw(31) << "Range";
    std::cout << std::endl;

    std::cout << std::setw(7) << "-------";
    std::cout << sep;
    std::cout << std::setw(15) << "---------------";
    std::cout << sep;
    std::cout << std::setw(10) << "----------";
    std::cout << sep;
    std::cout << std::setw(10) << "----------";
    std::cout << sep;
    std::cout << std::setw(10) << "----------";
    std::cout << sep;
    std::cout << std::setw(10) << "----------";
    std::cout << sep;
    std::cout << std::left << std::setw(31) << "-------------------------------";
    std::cout << std::endl;

    for (auto it = sorted_items.rbegin(); it != sorted_items.rend(); ++it) {
      const std::string& name = it->second;
      auto& item = items_.at(name);
      std::cout << std::right << std::setw(7) << (item.total_time / sum) * 100;
      std::cout << sep;
      std::cout << std::right << std::setw(15) << item.total_time;
      std::cout << sep;
      std::cout << std::right << std::setw(10) << item.instances;
      std::cout << sep;

      std::sort(item.elapsed.begin(), item.elapsed.end());
      std::cout << std::right << std::setw(10)
                // << item.total_time / static_cast<float>(item.instances);
                << item.elapsed[item.instances / 2];
      std::cout << sep;
      std::cout << std::right << std::setw(10) << item.minium;
      std::cout << sep;
      std::cout << std::right << std::setw(10) << item.maxium;
      std::cout << sep;
      std::cout << std::left << std::setw(31) << name;
      std::cout << std::endl;
    }
  }

 private:
  struct Frame {
    std::string name;
    uint64_t start_ns;
  };
  std::stack<Frame> stack_;

  struct TimeItem {
    uint64_t instances = 0;
    uint64_t total_time = 0;
    uint64_t minium = std::numeric_limits<uint64_t>::max();
    uint64_t maxium = 0;
    std::vector<uint64_t> elapsed;
  };
  std::unordered_map<std::string, TimeItem> items_;
};

static thread_local Profiler profiler;

void RangePush(const std::string& name) {
#ifdef OF_ENABLE_PROFILER
  // nvtxRangePushA(name.c_str());
  profiler.Push(name);
#endif  // OF_ENABLE_PROFILER
}

void RangePop() {
#ifdef OF_ENABLE_PROFILER
  // nvtxRangePop();
  profiler.Pop();
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

void EnableProfiler(bool use_cpu, bool use_cuda, bool record_shapes, bool record_bandwidth) {
  CHECK_JUST(vm::ClusterSync());
  if (Singleton<ProfileManager>::Get() == nullptr) {
    Singleton<ProfileManager>::New(use_cpu, use_cuda, record_shapes, record_bandwidth);
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
