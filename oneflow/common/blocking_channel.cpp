#include "common/blocking_channel.h"

namespace oneflow {

int BlockingChannel::Write(const T& val) {
  std::unique_lock<std::mutex> lck(mtx_);
  write_cond_.wait(lck);
  CHECK_NE(is_closed_, true);
  vals_.emplace(val);
  read_cond_.notify_one();
  return 0;
}

int BlockingChannel::Read(T* val) {
  std::unique_lock<std::mutex> lck(mtx_);
  read_cond_.wait(lck, [this](){ 
      return !this->vals_.empty() || this->is_closed_; });
  CHECK_NE(is_closed_, true);
  *val = vals_.front();
  vals_.pop();
  write_cond_.notify_one();
  return 0;
}

void BlockingChannel::Close() {
  std::unique_lock<std::mutex> lck(mtx_);
  is_closed_ = true;
  write_cond_.notify_all();
  read_cond_.notify_all();
}

}  // namespace oneflow
