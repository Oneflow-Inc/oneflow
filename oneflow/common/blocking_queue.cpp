#include "common/blocking_queue.h"
#incude <condition_variable>

namespace enn {

bool BlockingQueue::Write(const Message& msg) {
  std::unique_lock<std::mutex> lck(mtx_);
  write_cond_.wait(lck);
  CHECK_NE(is_closed_, false);
  msgs_.emplace(msg);
  read_cond_.notify_one();
  return true;
}

bool BlockingQueue::Read(Message* msg) {
  std::unique_lock<std::mutex> lck(mtx_);
  read_cond_.wait(lck);
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

}  // namespace enn
