#ifndef ONEFLOW_CORE_THREAD_THREAD_EXECUTOR_H_
#define ONEFLOW_CORE_THREAD_THREAD_EXECUTOR_H_

#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/thread/thread.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/platform/include/pthread_fork.h"

#ifdef WITH_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#endif

#ifdef WITH_OMP
#include <omp.h>
#endif

namespace oneflow {
namespace thread {

namespace {

template<typename F>
void SeqFor(int64_t begin, int64_t end, const F& func) {
  func(begin, end);
}

}  // namespace

class SeqExecutor;
class OfExecutor;

template<typename T>
class ExecutorBase {
 public:
  template<typename F>
  void ParallelFor(int64_t begin, int64_t end, const F& func, size_t num_threads,
                   size_t grain_size) {
    if (begin >= end) { return; }

    if (std::is_same<T, SeqExecutor>::value || num_threads == 1) {
      return SeqFor(begin, end, func);
    }
    if (std::is_same<T, OfExecutor>::value) {
      if (unlikely(pthread_fork::IsForkedSubProcess()) || Singleton<ThreadPool>::Get() == nullptr) {
        return SeqFor(begin, end, func);
      }
    }
    static_cast<T*>(this)->template ParallelFor(begin, end, func, num_threads, grain_size);
  }

 protected:
  static size_t DivUp(size_t x, size_t y) { return (x + y - 1) / y; }
};

class SeqExecutor final : public ExecutorBase<SeqExecutor> {};

class OfExecutor final : public ExecutorBase<OfExecutor> {
 public:
  template<typename F>
  static void ParallelFor(int64_t begin, int64_t end, const F& func, size_t num_threads,
                          size_t grain_size) {
    const size_t num_elements = end - begin;
    num_threads = std::min(num_elements, num_threads);
    BalancedSplitter bs(num_elements, num_threads);
    BlockingCounter bc(num_threads);

    FOR_RANGE(size_t, range_id, 0, num_threads) {
      Singleton<ThreadPool>::Get()->AddWork([&bc, &bs, range_id, func] {
        const size_t start_ = bs.At(range_id).begin();
        const size_t end_ = bs.At(range_id).end();
        func(start_, end_);
        bc.Decrease();
      });
    }
    // buzy loop wait.
    bc.WaitForeverUntilCntEqualZero();
  }
};

#if WITH_TBB
class TbbExecutor final : public ExecutorBase<TbbExecutor> {
 public:
  template<typename F>
  static void ParallelFor(int64_t begin, int64_t end, const F& func, size_t num_threads,
                          size_t grain_size) {
    tbb::global_control global_thread_limit(tbb::global_control::max_allowed_parallelism,
                                            num_threads);
    const size_t chunk_size = std::max(DivUp((end - begin), num_threads), grain_size);

    tbb::parallel_for(
        tbb::blocked_range<int64_t>(begin, end, chunk_size),
        [func](const tbb::blocked_range<int64_t>& r) { func(r.begin(), r.end()); },
        tbb::static_partitioner{});
  }
};
#endif

#if WITH_OMP
class OmpExecutor final : public ExecutorBase<OmpExecutor> {
 public:
  template<typename F>
  static void ParallelFor(int64_t begin, int64_t end, const F& func, size_t num_threads,
                          size_t grain_size) {
    num_threads = std::min(DivUp((end - begin), grain_size), num_threads);
#pragma omp parallel num_threads(num_threads)
    {
      int64_t omp_num_thread = omp_get_num_threads();
      int64_t chunk_size = DivUp((end - begin), omp_num_thread);
      int64_t omp_tid = omp_get_thread_num();
      int64_t thread_begin_index = begin + omp_tid * chunk_size;
      int64_t thread_end_index = std::min(end, chunk_size + thread_begin_index);

      if (thread_begin_index < end) { func(thread_begin_index, thread_end_index); }
    }
  }
};
#endif

}  // namespace thread
}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_EXECUTOR_H_
