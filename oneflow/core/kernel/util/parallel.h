#include "oneflow/core/framework/framework.h"

namespace oneflow {

inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// Called during new thread initialization
void init_num_threads();

// Sets the number of threads to be used in parallel region
void set_num_threads(int);

// Returns the maximum number of threads that may be used in a parallel region
int get_num_threads();

// Returns the current thread number (starting from 0)
// in the current parallel region, or 0 in the sequential region
int get_thread_num();

// Checks whether the code runs in parallel region
bool in_parallel_region();

namespace internal {

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define LIKELY(expr) (expr)
#define UNLIKELY(expr) (expr)
#endif

// Initialise num_threads lazily at first parallel call
inline void lazy_init_num_threads() {
  thread_local bool init = false;
  if (UNLIKELY(!init)) {
    oneflow::init_num_threads();
    init = true;
  }
}

} // namespace internal
} // namespace oneflow
