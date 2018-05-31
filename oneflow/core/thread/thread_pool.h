#ifndef ONEFLOW_CORE_THREAD_THREAD_POOL_H_
#define ONEFLOW_CORE_THREAD_THREAD_POOL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/channel.h"

namespace oneflow {

class ThreadPool final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ThreadPool);
  ThreadPool() = delete;
  ThreadPool(int32_t thread_num);
  ~ThreadPool();

  int32_t thread_num() const { return threads_.size(); }
  void AddWork(std::function<void()> work);

 private:
  std::vector<Channel<std::function<void()>>> work_chans_;
  std::vector<std::thread> threads_;

  std::mutex cur_chan_idx_mtx_;
  int32_t cur_chan_idx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_POOL_H_
