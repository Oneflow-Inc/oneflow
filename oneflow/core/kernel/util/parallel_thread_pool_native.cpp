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
#include "oneflow/core/kernel/util/parallel.h"
#include "oneflow/core/kernel/util/thread_pool.h"

namespace oneflow {

namespace {
const int NOT_SET = -1;
const int CONSUMED = -2;

// Number of inter-op threads set by the user;
// NOT_SET -> positive value -> CONSUMED
// (CONSUMED - thread pool is initialized)
// or
// NOT_SET -> CONSUMED
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<int> num_interop_threads{NOT_SET};

// thread pool global instance is hidden,
// users should use at::launch and get/set_num_interop_threads interface
internal::TaskThreadPoolBase& get_pool() {
  static std::shared_ptr<internal::TaskThreadPoolBase> pool =
      internal::ThreadPoolRegistry()->Create(
          "OneFlow",
          /* device_id */ 0,
          /* pool_size */ num_interop_threads.exchange(CONSUMED),
          /* create_new */ true);
  return *pool;
}

// Factory function for ThreadPoolRegistry
std::shared_ptr<internal::TaskThreadPoolBase> create_oneflow_threadpool(
    int device_id,
    int pool_size,
    bool create_new) {
  // For now, the only accepted device id is 0
  // TORCH_CHECK(device_id == 0);
  // Create new thread pool
  // TORCH_CHECK(create_new);
  return std::make_shared<internal::ThreadPool>(pool_size);
}

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
ONEFLOW_REGISTER_CREATOR(internal::ThreadPoolRegistry, OneFlow, create_oneflow_threadpool);

void set_num_interop_threads(int nthreads) {
  // TORCH_CHECK(nthreads > 0, "Expected positive number of threads");

  int no_value = NOT_SET;
  // TORCH_CHECK(num_interop_threads.compare_exchange_strong(no_value, nthreads),
  //     "Error: cannot set number of interop threads after parallel work "
  //     "has started or set_num_interop_threads called");
}

int get_num_interop_threads() {
  int nthreads = num_interop_threads.load();
  if (nthreads > 0) {
    return nthreads;
  } else if (nthreads == NOT_SET) {
    // return default value
    return internal::TaskThreadPoolBase::defaultNumThreads();
  } else {
    return get_pool().size();
  }
}

namespace internal {
void launch_no_thread_state(std::function<void()> fn) {
#if AT_EXPERIMENTAL_SINGLE_THREAD_POOL
  intraop_launch(std::move(fn));
#else
  get_pool().run(std::move(fn));
#endif
}
} // namespace internal

void launch(std::function<void()> func) {
  // NOLINTNEXTLINE(modernize-avoid-bind)
  internal::launch_no_thread_state(std::bind([](
    std::function<void()> f) {
      // NOLINTNEXTLINE(performance-move-const-arg)
      f();
    },
    std::move(func)
  ));
}


}  // namespace oneflow
