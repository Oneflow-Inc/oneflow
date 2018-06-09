#include "oneflow/core/thread/thread_pool.h"

namespace oneflow {

ThreadPool::ThreadPool(int32_t thread_num)
    : work_chans_(thread_num), threads_(thread_num), cur_chan_idx_(0) {
  FOR_RANGE(int32_t, i, 0, thread_num) {
    Channel<std::function<void()>>* chan = &(work_chans_.at(i));
    threads_[i] = std::thread([chan]() {
      std::function<void()> work;
      while (chan->Receive(&work) == 0) { work(); }
    });
  }
}

ThreadPool::~ThreadPool() {
  FOR_RANGE(int32_t, i, 0, work_chans_.size()) {
    work_chans_.at(i).CloseSendEnd();
    work_chans_.at(i).CloseReceiveEnd();
    threads_.at(i).join();
  }
}

void ThreadPool::AddWork(std::function<void()> work) {
  if (work_chans_.size() > 1) {
    std::unique_lock<std::mutex> lck(cur_chan_idx_mtx_);
    work_chans_.at(cur_chan_idx_).Send(work);
    cur_chan_idx_ = (cur_chan_idx_ + 1) % work_chans_.size();
  } else {
    work_chans_.at(cur_chan_idx_).Send(work);
  }
}

}  // namespace oneflow
