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

#define OF_RUNTIME_SEQ 0u
#define OF_RUNTIME_OMP 1u
#define OF_RUNTIME_TBB 2u

#if OF_CPU_THREADING_RUNTIME == OF_RUNTIME_OMP
#include <omp.h>
#elif OF_CPU_THREADING_RUNTIME == OF_RUNTIME_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#endif

#ifdef WITH_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace oneflow {

namespace ep {

class CpuStream : public Stream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuStream);
  explicit CpuStream(Device* device) : device_(device) {
#ifdef WITH_ONEDNN
    onednn_engine_.reset(new dnnl::engine(dnnl::engine::kind::cpu, 0));
    onednn_stream_.reset(new dnnl::stream(*onednn_engine_));
#endif
  }

  ~CpuStream() override = default;

  DeviceType device_type() const override;
  Device* device() const override;
  Maybe<void> Sync() override;
  void RecordEvent(Event* event) override;

  void SetParallelNumberThreads(size_t num_threads) {
  #if OF_CPU_THREADING_RUNTIME == OF_RUNTIME_OMP
    // Affects omp_get_max_threads() Get the logical core book
    omp_set_num_threads(number_threads);
  #elif OF_CPU_THREADING_RUNTIME == OF_RUNTIME_TBB
    tbb::global_control global_thread_limit(tbb::global_control::max_allowed_parallelism, num_threads);
  #endif
  }

  template<typename F>
  void Parallel(int64_t begin, int64_t end, const F& func, size_t grain_size, size_t num_threads) {
    auto divup = [] (int64_t x, int64_t y) { return (x + y - 1) / y; };

    if (begin >= end) { return; }
  #if OF_CPU_THREADING_RUNTIME == OF_RUNTIME_OMP
    if (grain_size > 0) { num_threads = std::min(num_threads, divup((end - begin), grain_size)); }
  #pragma omp parallel num_threads(num_threads)
    {
      int64_t chunk_size = divup((end - begin), num_threads);
      int64_t omp_tid = omp_get_thread_num();
      int64_t thread_begin_index = begin + omp_tid * chunk_size;
      int64_t thread_end_index = std::min(end, chunk_size + thread_begin_index);

      if (thread_begin_index < end) { func(thread_begin_index, thread_end_index); }
    }

  #elif OF_CPU_THREADING_RUNTIME == OF_RUNTIME_TBB
    SetParallelNumberThreads(num_threads);
    size_t nthr_chunk_size = divup((end - begin), num_threads);
    int64_t chunk_size = std::max(nthr_chunk_size, grain_size);

    tbb::parallel_for(
        tbb::blocked_range<int64_t>(begin, end, chunk_size),
        [func](const tbb::blocked_range<int64_t>& r) { func(r.begin(), r.end()); },
        tbb::static_partitioner{});
  #else
    func(begin, end);
  #endif
  }

#ifdef WITH_ONEDNN
  dnnl::engine* onednn_engine() const { return onednn_engine_.get(); }
  dnnl::stream* onednn_stream() const { return onednn_stream_.get(); }

 private:
  std::unique_ptr<dnnl::engine> onednn_engine_;
  std::unique_ptr<dnnl::stream> onednn_stream_;
#endif
  Device* device_;
};

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_CPU_CPU_STREAM_H_
