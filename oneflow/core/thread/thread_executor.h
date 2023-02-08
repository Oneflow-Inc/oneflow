#ifndef ONEFLOW_CORE_THREAD_THREAD_EXECUTOR_H_
#define ONEFLOW_CORE_THREAD_THREAD_EXECUTOR_H_

#include <functional>
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

using CallableT = std::function<void(int64_t, int64_t)>;

void SeqFor(int64_t begin, int64_t end, const CallableT& func) { func(begin, end); }

size_t DivUp(size_t x, size_t y) { return (x + y - 1) / y; }

}  // namespace

class ExecutorBase {
 public:
  virtual void ParallelFor(int64_t begin, int64_t end, const CallableT& func, size_t num_threads,
                           size_t grain_size) {
    MayRunSeq(begin, end, func, num_threads);
  }

  bool MayRunSeq(int64_t begin, int64_t end, const CallableT& func, size_t num_threads) {
    if (begin >= end) { return true; }
    if (num_threads == 1) {
      SeqFor(begin, end, func);
      return true;
    }
    return false;
  }
};

class SeqExecutor final : public ExecutorBase {};

class OfExecutor final : public ExecutorBase {
 public:
  void ParallelFor(int64_t begin, int64_t end, const CallableT& func, size_t num_threads,
                   size_t grain_size) override {
    if (MayRunSeq(begin, end, func, num_threads)) { return; }
    if (unlikely(pthread_fork::IsForkedSubProcess()) || Singleton<ThreadPool>::Get() == nullptr) {
      return SeqFor(begin, end, func);
    }
    const size_t num_elements = end - begin;
    num_threads = std::min(num_elements, num_threads);
    BalancedSplitter bs(num_elements, num_threads);
    BlockingCounter bc(num_threads);

    FOR_RANGE(size_t, range_id, 0, num_threads) {
      Singleton<ThreadPool>::Get()->AddWork([&bc, &bs, range_id, func] {
        const size_t begin_ = bs.At(range_id).begin();
        const size_t end_ = bs.At(range_id).end();
        SeqFor(begin_, end_, func);
        bc.Decrease();
      });
    }
    // buzy loop wait.
    bc.WaitForeverUntilCntEqualZero();
  }
};

#if WITH_TBB
class TbbExecutor final : public ExecutorBase {
 public:
  void ParallelFor(int64_t begin, int64_t end, const CallableT& func, size_t num_threads,
                   size_t grain_size) override {
    if (MayRunSeq(begin, end, func, num_threads)) { return; }
    tbb::global_control global_thread_limit(tbb::global_control::max_allowed_parallelism,
                                            num_threads);
    const size_t chunk_size = std::max(DivUp((end - begin), num_threads), grain_size);

    tbb::parallel_for(
        tbb::blocked_range<int64_t>(begin, end, chunk_size),
        [&func](const tbb::blocked_range<int64_t>& r) { SeqFor(r.begin(), r.end(), func); },
        tbb::static_partitioner{});
  }
};
#endif

#if WITH_OMP
class OmpExecutor final : public ExecutorBase {
 public:
  void ParallelFor(int64_t begin, int64_t end, const CallableT& func, size_t num_threads,
                   size_t grain_size) override {
    if (MayRunSeq(begin, end, func, num_threads)) { return; }
    num_threads = std::min(DivUp((end - begin), grain_size), num_threads);
#pragma omp parallel num_threads(num_threads)
    {
      int64_t omp_num_thread = omp_get_num_threads();
      int64_t chunk_size = DivUp((end - begin), omp_num_thread);
      int64_t omp_tid = omp_get_thread_num();
      int64_t thread_begin_index = begin + omp_tid * chunk_size;
      int64_t thread_end_index = std::min(end, chunk_size + thread_begin_index);

      if (thread_begin_index < end) { SeqFor(thread_begin_index, thread_end_index, func); }
    }
  }
};
#endif

}  // namespace thread
}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_EXECUTOR_H_
