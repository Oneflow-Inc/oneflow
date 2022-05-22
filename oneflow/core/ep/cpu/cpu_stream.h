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
#ifndef ONEFLOW_CORE_EP_CPU_CPU_STREAM_H_
#define ONEFLOW_CORE_EP_CPU_CPU_STREAM_H_

#include <functional>
#include <memory>
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"

#define OF_RUNTIME_SEQ 0u
#define OF_RUNTIME_OMP 1u
#define OF_RUNTIME_TBB 2u

#if OF_CPU_THREADING_RUNTIME == OF_RUNTIME_OMP
#include <omp.h>
#elif OF_CPU_THREADING_RUNTIME == OF_RUNTIME_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#elif OF_CPU_THREADING_RUNTIME == OF_RUNTIME_SEQ
// Nothing
#else
#error OF_CPU_THREADING_RUNTIME Error setting
#endif

#ifdef WITH_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace oneflow {

namespace ep {

class CpuNumThreadsGuard {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuNumThreadsGuard);
#if OF_CPU_THREADING_RUNTIME == OF_RUNTIME_TBB
  explicit CpuNumThreadsGuard(size_t num_threads)
      : global_thread_limit(tbb::global_control::max_allowed_parallelism, num_threads) {}
  ~CpuNumThreadsGuard() {}
#elif OF_CPU_THREADING_RUNTIME == OF_RUNTIME_OMP
  explicit CpuNumThreadsGuard(size_t num_threads) : set_num_threads_(num_threads) {
    saved_num_threads_ = omp_get_max_threads();
    omp_set_num_threads(set_num_threads_);
  }
  ~CpuNumThreadsGuard() { omp_set_num_threads(saved_num_threads_); }

#elif OF_CPU_THREADING_RUNTIME == OF_RUNTIME_SEQ
  explicit CpuNumThreadsGuard(size_t num_threads) {}
  ~CpuNumThreadsGuard() {}
#else
#error OF_CPU_THREADING_RUNTIME Error setting
#endif

 private:
#if OF_CPU_THREADING_RUNTIME == OF_RUNTIME_TBB
  tbb::global_control global_thread_limit;
#elif OF_CPU_THREADING_RUNTIME == OF_RUNTIME_OMP
  size_t set_num_threads_;
  size_t saved_num_threads_;
#elif OF_CPU_THREADING_RUNTIME == OF_RUNTIME_SEQ

#else
#error OF_CPU_THREADING_RUNTIME Error setting
#endif
};

#ifdef WITH_ONEDNN

class OneDNNFallback;

#endif

class CpuStream : public Stream {
 public:
  using ParallelForFuncType = std::function<void(int64_t, int64_t)>;

  OF_DISALLOW_COPY_AND_MOVE(CpuStream);

  explicit CpuStream(CpuDevice* device) : device_(device) {}

  ~CpuStream() override = default;

  DeviceType device_type() const override;
  CpuDevice* device() const override;
  Maybe<void> Sync() override;
  void RecordEvent(Event* event) override;

  void ParallelFor(int64_t begin, int64_t end, const ParallelForFuncType& func);
  void ParallelFor(int64_t begin, int64_t end, const ParallelForFuncType& func, size_t grain_size);

#ifdef WITH_ONEDNN
  const std::unique_ptr<ep::OneDNNFallback>& onednn_fallback() const;
#endif

 private:
#ifdef WITH_ONEDNN
  std::unique_ptr<ep::OneDNNFallback> onednn_fallback_ = std::make_unique<ep::OneDNNFallback>(this);
#endif
  CpuDevice* device_;
  static constexpr size_t kParallelForDefaultGrain = 32768;
};

#ifdef WITH_ONEDNN

class OneDNNFallback {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OneDNNFallback);

  OneDNNFallback() = delete;

  explicit OneDNNFallback(CpuStream* cpu_stream) : cpu_stream_(cpu_stream) {
    engine_.reset(new dnnl::engine(dnnl::engine::kind::cpu, 0));
    stream_.reset(new dnnl::stream(*engine_));
  }

  ~OneDNNFallback() = default;

  void Launch(const std::function<void(dnnl::engine*, dnnl::stream*)>& f);

 private:
  CpuStream* cpu_stream_ = nullptr;
  std::unique_ptr<dnnl::engine> engine_;
  std::unique_ptr<dnnl::stream> stream_;
};

#endif

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_CPU_CPU_STREAM_H_
