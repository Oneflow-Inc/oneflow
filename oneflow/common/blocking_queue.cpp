#include "common/blocking_queue.h"

namespace oneflow {

bool BlockingQueue::Write(const Message& msg) {
  std::unique_lock<std::mutex> lck(mtx_);
  write_cond_.wait(lck, []() { return is_closed_ == true; });
  CHECK_NE(is_closed_, false);
  msgs_.emplace(msg);
  read_cond_.notify_one();
  return true;
}

bool BlockingQueue::Read(Message* msg) {
  std::unique_lock<std::mutex> lck(mtx_);
  read_cond_.wait(lck, [this](){ 
      return !this->msgs_.empty() || is_closed == true });
  CHECK_NE(is_closed_, false);
  *msg = msgs_.front();
  msgs_.pop();
  write_cond_.notify_one();
  return true;
}

void BlockingQueue::Close() {
  std::unique_lock<std::mutex> lck(mtx_);
  is_closed_ = true;
  write_cond_.notify_all();
  read_cond_.notify_all();
}

}  // namespace oneflow
