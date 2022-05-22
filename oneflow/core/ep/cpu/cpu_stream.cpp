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
#include "oneflow/core/ep/cpu/cpu_stream.h"

namespace oneflow {

namespace ep {

DeviceType CpuStream::device_type() const { return DeviceType::kCPU; }

CpuDevice* CpuStream::device() const { return device_; }

Maybe<void> CpuStream::Sync() { return Maybe<void>::Ok(); }

void CpuStream::RecordEvent(Event* /*event*/) {}

void CpuStream::ParallelFor(int64_t begin, int64_t end, const ParallelForFuncType& func) {
  ParallelFor(begin, end, func, kParallelForDefaultGrain);
}

void CpuStream::ParallelFor(int64_t begin, int64_t end, const ParallelForFuncType& func,
                            size_t grain_size) {
#if OF_CPU_THREADING_RUNTIME != OF_RUNTIME_SEQ
  auto DivUp = [](int64_t x, int64_t y) { return (x + y - 1) / y; };
  size_t num_threads = device()->GetNumThreads();
#endif
  if (begin >= end) { return; }
#if OF_CPU_THREADING_RUNTIME == OF_RUNTIME_OMP
  if (grain_size > 0) {
    num_threads = std::min(num_threads, (size_t)(DivUp((end - begin), grain_size)));
  } else {
    num_threads = 1;
  }
#pragma omp parallel num_threads(num_threads)
  {
    int64_t omp_num_thread = omp_get_num_threads();
    int64_t chunk_size = DivUp((end - begin), omp_num_thread);
    int64_t omp_tid = omp_get_thread_num();
    int64_t thread_begin_index = begin + omp_tid * chunk_size;
    int64_t thread_end_index = std::min(end, chunk_size + thread_begin_index);

    if (thread_begin_index < end) { func(thread_begin_index, thread_end_index); }
  }

#elif OF_CPU_THREADING_RUNTIME == OF_RUNTIME_TBB
  CpuNumThreadsGuard guard(num_threads);
  size_t tmp_chunk_size = DivUp((end - begin), num_threads);
  int64_t chunk_size = std::max(tmp_chunk_size, grain_size);

  tbb::parallel_for(
      tbb::blocked_range<int64_t>(begin, end, chunk_size),
      [func](const tbb::blocked_range<int64_t>& r) { func(r.begin(), r.end()); },
      tbb::static_partitioner{});
#elif OF_CPU_THREADING_RUNTIME == OF_RUNTIME_SEQ
  func(begin, end);
#else
#error OF_CPU_THREADING_RUNTIME Error setting
#endif
}

#ifdef WITH_ONEDNN

const std::unique_ptr<ep::OneDNNFallback>& CpuStream::onednn_fallback() const {
  return onednn_fallback_;
}

void OneDNNFallback::Launch(const std::function<void(dnnl::engine*, dnnl::stream*)>& f) {
  CpuNumThreadsGuard guard(cpu_stream_->device()->GetNumThreads());
  f(engine_.get(), stream_.get());
  stream_->wait();
}

#endif

}  // namespace ep

}  // namespace oneflow
