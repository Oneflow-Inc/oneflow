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
#include <algorithm>
#include <cstddef>
#include <exception>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/parallel.h"

namespace oneflow {
namespace internal {

inline std::tuple<size_t, size_t> calc_num_tasks_and_chunk_size(int64_t begin, int64_t end,
                                                                int64_t grain_size) {
  if ((end - begin) < grain_size) { return std::make_tuple(1, std::max((int64_t)0, end - begin)); }
  // Choose number of tasks based on grain size and number of threads.
  size_t chunk_size = divup((end - begin), get_num_threads());
  // Make sure each task is at least grain_size size.
  chunk_size = std::max((size_t)grain_size, chunk_size);
  size_t num_tasks = divup((end - begin), chunk_size);
  printf("\n parallel_native.h >>> calc_num_tasks_and_chunk_size() >>> begin:%ld; end:%ld; grain_size:%ld; chunk_size:%zu, num_tasks:%zu", 
    begin, end, grain_size, chunk_size, num_tasks
  );
  return std::make_tuple(num_tasks, chunk_size);
}

void _parallel_run(const int64_t begin, const int64_t end, const int64_t grain_size,
                   const std::function<void(int64_t, int64_t, size_t)>& f);

}  // namespace internal

template<class F>
inline void parallel_for(const int64_t begin, const int64_t end, const int64_t grain_size,
                         const F& f) {
  CHECK_GE(grain_size, 0);
  printf("\n===============parallel_native.h >>> parallel_for()================");
  if (begin >= end) { return; }
  if ((end - begin) < grain_size || in_parallel_region()) {
    f(begin, end);
    return;
  }
  internal::_parallel_run(begin, end, grain_size,
                          [f](int64_t start, int64_t end, size_t /* unused */) { f(start, end); });
}

}  // namespace oneflow
