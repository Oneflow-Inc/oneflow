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
#ifndef ONEFLOW_CORE_EP_PARALLEL_H_
#define ONEFLOW_CORE_EP_PARALLEL_H_
#include <iostream>
#include <unistd.h>
#include <sys/types.h>

#define OF_RUNTIME_OMP 1u
#define OF_RUNTIME_TBB 2u

#if WITH_CPU_THREADING_RUNTIME == OF_RUNTIME_OMP
#include <omp.h>
#elif WITH_CPU_THREADING_RUNTIME == OF_RUNTIME_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif
namespace oneflow {
namespace ep {
namespace primitive {

inline size_t divup(int64_t x, int64_t y) { return (x + y - 1) / y; }

template<typename F>
void parallel(int64_t begin, int64_t end, const F& func, size_t grain_size, size_t nthr) {
  if (begin >= end) { return; }

#if WITH_CPU_THREADING_RUNTIME == OF_RUNTIME_OMP

  if (grain_size > 0) { nthr = std::min(nthr, divup((end - begin), grain_size)); }
#pragma omp parallel num_threads(nthr)
  {
    int64_t chunk_size = divup((end - begin), nthr);
    int64_t tid = omp_get_thread_num();
    int64_t begin_tid = begin + tid * chunk_size;
    int64_t end_tid = std::min(end, chunk_size + begin_tid);

    if (begin_tid < end) { func(begin_tid, end_tid); }
  }

#elif WITH_CPU_THREADING_RUNTIME == OF_RUNTIME_TBB
  int64_t chunk_size = divup((end - begin), nthr);
  chunk_size = std::max(grain_size, chunk_size);
  tbb::parallel_for(tbb::blocked_range<int64_t>(begin, end, chunk_size),
    [f](const tbb::blocked_range<int64_t>& r) {
        f(r.begin(), r.end());
    }, tbb::static_partitioner{});
#else
  func(begin, end);
#endif
}

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
#endif  // ONEFLOW_CORE_EP_EVENT_H_
