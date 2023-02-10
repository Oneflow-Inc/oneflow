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

#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"
#include "oneflow/core/thread/thread_runtime_factory.h"

#ifdef WITH_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace oneflow {

namespace ep {

class CpuNumThreadsGuard {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuNumThreadsGuard);
#if WITH_TBB
  explicit CpuNumThreadsGuard(size_t num_threads)
      : global_thread_limit(tbb::global_control::max_allowed_parallelism, num_threads) {}
  ~CpuNumThreadsGuard() {}
#elif WITH_OMP
  explicit CpuNumThreadsGuard(size_t num_threads) : set_num_threads_(num_threads) {
    saved_num_threads_ = omp_get_max_threads();
    omp_set_num_threads(set_num_threads_);
  }
  ~CpuNumThreadsGuard() { omp_set_num_threads(saved_num_threads_); }
#endif

 private:
#if WITH_TBB
  tbb::global_control global_thread_limit;
#elif WITH_OMP
  size_t set_num_threads_;
  size_t saved_num_threads_;
#endif
};

#ifdef WITH_ONEDNN

class OneDnnExecutor;

#endif

class CpuStream : public Stream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuStream);

  explicit CpuStream(CpuDevice* device) : device_(device) {
    CHECK_JUST(InitThreadRuntime());
#ifdef WITH_ONEDNN
    onednn_executor_ = std::make_unique<ep::OneDnnExecutor>(this);
#endif
  }

  ~CpuStream() override = default;

  DeviceType device_type() const override;
  CpuDevice* device() const override;
  Maybe<void> Sync() override;
  void RecordEvent(Event* event) override;

  template<typename F>
  void ParallelFor(int64_t begin, int64_t end, const F& func) {
    ParallelFor(begin, end, func, kParallelForDefaultGrain);
  }

  template<typename F>
  void ParallelFor(int64_t begin, int64_t end, const F& func, size_t grain_size) {
    thread_runtime_->ParallelFor(begin, end, func, device()->GetNumThreads(), grain_size);
  }

#ifdef WITH_ONEDNN
  const std::unique_ptr<ep::OneDnnExecutor>& onednn_executor() const;
#endif

 private:
  CpuDevice* device_;
  static constexpr size_t kParallelForDefaultGrain = 32768;
  std::shared_ptr<thread::RuntimeBase> thread_runtime_;

  Maybe<void> InitThreadRuntime();

#ifdef WITH_ONEDNN
  std::unique_ptr<ep::OneDnnExecutor> onednn_executor_;
#endif
};

#ifdef WITH_ONEDNN

class OneDnnExecutor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OneDnnExecutor);

  OneDnnExecutor() = delete;

  explicit OneDnnExecutor(CpuStream* cpu_stream) : cpu_stream_(cpu_stream) {
    engine_.reset(new dnnl::engine(dnnl::engine::kind::cpu, 0));
    stream_.reset(new dnnl::stream(*engine_));
  }

  ~OneDnnExecutor() = default;

  template<typename F>
  void Launch(const F& f) {
    CpuNumThreadsGuard guard(cpu_stream_->device()->GetNumThreads());
    f(engine_.get(), stream_.get());
    stream_->wait();
  }

 private:
  CpuStream* cpu_stream_ = nullptr;
  std::unique_ptr<dnnl::engine> engine_;
  std::unique_ptr<dnnl::stream> stream_;
};

#endif

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_CPU_CPU_STREAM_H_
