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
