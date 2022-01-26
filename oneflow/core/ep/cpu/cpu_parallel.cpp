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
#include "oneflow/core/ep/cpu/cpu_parallel.h"

namespace oneflow {

namespace ep {

void set_num_threads(int nthr) {
#if OF_CPU_THREADING_RUNTIME == OF_RUNTIME_OMP
  // Affects omp_get_max_threads() Get the logical core book
  omp_set_num_threads(nthr);
#elif OF_CPU_THREADING_RUNTIME == OF_RUNTIME_TBB
  tbb::global_control global_thread_limit(tbb::global_control::max_allowed_parallelism, nthr);
#endif
}

}  // namespace ep
}  // namespace oneflow
