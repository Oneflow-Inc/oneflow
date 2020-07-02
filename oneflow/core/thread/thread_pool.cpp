#include "oneflow/core/thread/thread_pool.h"

namespace oneflow {

ThreadPool::ThreadPool(int32_t thread_num)
    : work_chans_(thread_num), threads_(thread_num), work_cnt_(0) {
  FOR_RANGE(int32_t, i, 0, thread_num) {
    Channel<std::function<void()>>* chan = &(work_chans_.at(i));
    threads_[i] = std::thread([chan]() {
      std::function<void()> work;
      while (chan->Receive(&work) == kChannelStatusSuccess) { work(); }
    });
  }
}

ThreadPool::~ThreadPool() {
  FOR_RANGE(int32_t, i, 0, work_chans_.size()) {
    work_chans_.at(i).Close();
    threads_.at(i).join();
  }
}

void ThreadPool::AddWork(const std::function<void()>& work) {
  const size_t cur_chan_idx =
      work_cnt_.fetch_add(1, std::memory_order_relaxed) % work_chans_.size();
  work_chans_.at(cur_chan_idx).Send(work);
}

}  // namespace oneflow
