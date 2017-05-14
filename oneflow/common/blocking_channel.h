#ifndef ONEFLOW_COMMON_BLOCKING_CHANNEL_H_
#define ONEFLOW_COMMON_BLOCKING_CHANNEL_H_

#include <queue>
#include <mutex>
#include <condition_variable>
#include "common/util.h"

namespace oneflow {

template<typename T>
class BlockingChannel final {
public:
  OF_DISALLOW_COPY_AND_MOVE(BlockingChannel);
  BlockingChannel() = default;
  ~BlockingChannel() = default;

  int Write(const T& val) {
    std::unique_lock<std::mutex> lck(mtx_);
    write_cond_.wait(lck);
    CHECK_NE(is_closed_, true);
    vals_.emplace(val);
    read_cond_.notify_one();
    return 0;
  }

  int Read(T* val) {
    std::unique_lock<std::mutex> lck(mtx_);
    read_cond_.wait(lck, [this](){ 
        return !this->vals_.empty() || this->is_closed_; });
    CHECK_NE(is_closed_, true);
    *val = vals_.front();
    vals_.pop();
    write_cond_.notify_one();
    return 0;
  }

  void Close() {
    std::unique_lock<std::mutex> lck(mtx_);
    is_closed_ = true;
    write_cond_.notify_all();
    read_cond_.notify_all();
  }

private:
  std::queue<T> vals_;
  std::mutex mtx_;
  std::condition_variable write_cond_;
  std::condition_variable read_cond_;
  bool is_closed_;
};

}  // namespace oneflow

#endif  // ONEFLOW_COMMON_BLOCKING_CHANNEL_H_
