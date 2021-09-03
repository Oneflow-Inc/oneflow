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
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/thread_pool.h"
#include "oneflow/core/kernel/util/parallel_native.h"

namespace oneflow {
namespace {

inline int stoi(const std::string& str, std::size_t* pos = 0) {
  std::stringstream ss;
  int n = 0;
  ss << str;
  ss >> n;
  if (ss.fail()) {
    // To mimic `std::stoi` and to avoid including `Exception.h`, throw
    // `std::invalid_argument`.
    // We can't easily detect out-of-range, so we don't use `std::out_of_range`.
    throw std::invalid_argument("Not an integer");
  }
  if (pos) {
    if (ss.tellg() == std::streampos(-1)) {
      *pos = str.size();
    } else {
      *pos = ss.tellg();
    }
  }
  return n;
}

size_t get_env_num_threads(const char* var_name, size_t def_value = 0) {
  try {
    if (auto* value = std::getenv(var_name)) {
      int nthreads = stoi(value);
      CHECK_GT(nthreads, 0);
      return nthreads;
    }
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "Invalid " << var_name << " variable value, " << e.what();
    LOG(WARNING) << oss.str();
  }
  return def_value;
}

int intraop_default_num_threads() {
  size_t nthreads = get_env_num_threads("OMP_NUM_THREADS", 0);
  nthreads = get_env_num_threads("MKL_NUM_THREADS", nthreads);
  if (nthreads == 0) { nthreads = oneflow::internal::TaskThreadPoolBase::defaultNumThreads(); }
  return nthreads;
}

// used with _set_in_parallel_region to mark master thread
// as in parallel region while executing parallel primitives
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local bool in_parallel_region_ = false;

// thread number (task_id) set by parallel primitive
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local size_t thread_num_ = 0;

void _set_in_parallel_region(bool in_region) { in_parallel_region_ = in_region; }

void _set_thread_num(size_t thread_num) { thread_num_ = thread_num; }

void _unset_thread_num() { thread_num_ = 0; }

const int NOT_SET = -1;
const int CONSUMED = -2;

// Number of threads set by the user
// NOT_SET -> positive value -> CONSUMED
// or
// NOT_SET -> CONSUMED
// Meaning:
//  - NOT_SET - pool not initialized, user value is not set
//  - positive value - pool not initialized, user value set
//  - CONSUMED - pool is initialized
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<int> num_intraop_threads{NOT_SET};

int _num_pool_threads(int nthreads) {
  printf("\nparallel_native.cpp >>> _num_pool_threads() >>> input nthreads %d", nthreads);
  if (nthreads == NOT_SET) {
    nthreads = intraop_default_num_threads();
  } else {
    CHECK_GT(nthreads, 0);
  }
  // minus one because of the master thread
  printf("\nparallel_native.cpp >>> _num_pool_threads() >>> return nthreads %d", nthreads);
  return nthreads - 1;
}

oneflow::internal::TaskThreadPoolBase& _get_intraop_pool() {
  printf("\nparallel_native.cpp >>> oneflow::internal::TaskThreadPoolBase& >>>  _get_intraop_pool()");
  static std::shared_ptr<oneflow::internal::TaskThreadPoolBase> pool =
      oneflow::internal::ThreadPoolRegistry()->Create(
          "OneFlow",
          /* device_id */ 0,
          /* pool_size */ _num_pool_threads(num_intraop_threads.exchange(CONSUMED)),
          /* create_new */ true);  // create a separate thread pool for intra-op
  
  printf("\nparallel_native.cpp >>> oneflow::internal::TaskThreadPoolBase& >>>  _get_intraop_pool() >> create success!");
  return *pool;
}

// Run lambda function `fn` over `task_id` in [0, `range`) with threadpool.
// `fn` will be called with params: (thread_pool_task_id, task_id).
void _run_with_pool(const std::function<void(int, size_t)>& fn, size_t range) {
  printf("\nparallel_native.cpp >>>>>>>>> _run_with_pool() >>> range:%zu", range);
  for (size_t i = 1; i < range; ++i) {
    printf("\nparallel_native.cpp >>>>>>>>> _run_with_pool() >>> i:%zu", i);
    _get_intraop_pool().run([fn, i]() { fn((int)i, i); });
  }
  // Run the first task on the current thread directly.
  fn(0, 0);
}

// RAII guard helps to support in_parallel_region() and get_thread_num() API.
struct ParallelRegionGuard {
  ParallelRegionGuard(int64_t task_id) {
    _set_thread_num(task_id);
    _set_in_parallel_region(true);
  }

  ~ParallelRegionGuard() {
    _set_in_parallel_region(false);
    _unset_thread_num();
  }
};

}  // namespace
namespace internal {

void _parallel_run(const int64_t begin, const int64_t end, const int64_t grain_size,
                   const std::function<void(int64_t, int64_t, size_t)>& f) {
  oneflow::internal::lazy_init_num_threads();
  printf("\n===============parallel_native.cpp >>> _parallel_run()================");
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t num_tasks, chunk_size;
  std::tie(num_tasks, chunk_size) = internal::calc_num_tasks_and_chunk_size(begin, end, grain_size);

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  struct {
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
    std::mutex mutex;
    volatile size_t remaining;
    std::condition_variable cv;
  } state;

  printf("\nparallel_native.cpp >>>>>>>>> task\n");

  auto task = [f, &state, begin, end, chunk_size](int /* unused */, size_t task_id) {
    int64_t local_start = begin + task_id * chunk_size;
    printf("\nparallel_native.cpp >>>>>>>>> ParallelRegionGuard() >>> local_start:%ld; task_id:%zu \n", local_start, task_id);
    if (local_start < end) {
      int64_t local_end = std::min(end, (int64_t)(chunk_size + local_start));
      try {
        printf("\nparallel_native.cpp >>>>>>>>> ParallelRegionGuard() >>> tyr() ");
        ParallelRegionGuard guard(task_id);
        f(local_start, local_end, task_id);
      } catch (...) {
        if (!state.err_flag.test_and_set()) { state.eptr = std::current_exception(); }
      }
    }
    {
      std::unique_lock<std::mutex> lk(state.mutex);
      if (--state.remaining == 0) { state.cv.notify_one(); }
    }
  };
  printf("\nparallel_native.cpp >>>>>>>>> state.remaining = num_tasks >>> num_tasks: %zu", num_tasks);
  state.remaining = num_tasks;
  printf("\nparallel_native.cpp >>>>>>>>> _run_with_pool(task, num_tasks);");
  _run_with_pool(task, num_tasks);

  // Wait for all tasks to finish.
  {
    std::unique_lock<std::mutex> lk(state.mutex);
    if (state.remaining != 0) { state.cv.wait(lk); }
  }
  if (state.eptr) { std::rethrow_exception(state.eptr); }
}

}  // namespace internal

void init_num_threads() {
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
}

void set_num_threads(int nthreads) {
  printf("\nparallel_native.cpp >>>>>>>>>  set_num_threads() >>> %d\n", nthreads);
  // TORCH_CHECK(nthreads > 0, "Expected positive number of threads");
  int no_value = NOT_SET;
  if (!num_intraop_threads.compare_exchange_strong(no_value, nthreads)) {
    // num_intraop_threads either stores a positive integer or CONSUMED,
    // check that requested size is the same as the current one
    int stored_nthreads = num_intraop_threads.load();
    if (stored_nthreads <= 0) {
      // plus one because of master thread
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      stored_nthreads = _get_intraop_pool().size() + 1;
    }
    if (stored_nthreads != nthreads) {
      // TORCH_WARN(
      LOG(WARNING) << "Cannot set number of intraop threads "
                   << "after parallel work has started or after set_num_threads call "
                   << "when using native parallel backend";
    }
  }
}

int get_num_threads() {
  // not initializing pool unnecessarily,
  // because pool cannot be resized after initialization
  int nthreads = num_intraop_threads.load();
  printf("\nparallel_native.cpp >>>>>>>>>  get_num_threads() >>> nthreads:%d", nthreads);
  if (nthreads > 0) {
    return nthreads;
  } else if (nthreads == NOT_SET) {
    return intraop_default_num_threads();
  } else {
    CHECK_EQ(nthreads, CONSUMED);
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    return _get_intraop_pool().size() + 1;
  }
}

int get_thread_num() { return thread_num_; }

bool in_parallel_region() {
  return in_parallel_region_
         || (num_intraop_threads.load() == CONSUMED &&
             // Needed as intraop_launch() doesn't set in_parallel_region().
             _get_intraop_pool().inThreadPool());
}

void intraop_launch(std::function<void()> func) {
  if (!in_parallel_region() && get_num_threads() > 1) {
    _get_intraop_pool().run(func);
  } else {
    // execute inline if we're in parallel region
    func();
  }
}

}  // namespace oneflow
