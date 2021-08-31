#include "oneflow/core/common/notifier.h"

namespace oneflow {

NotifierStatus Notifier::Notify() {
  bool notify;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (is_closed_) { return kNotifierStatusErrorClosed; }
    notify = (notified_cnt_ == 0);
    ++notified_cnt_;
  }
  if (notify) { cond_.notify_one(); }
  return kNotifierStatusSuccess;
}

NotifierStatus Notifier::WaitAndClearNotifiedCnt() {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this]() { return notified_cnt_ > 0 || is_closed_; });
  if (notified_cnt_ == 0) { return kNotifierStatusErrorClosed; }
  notified_cnt_ = 0;
  return kNotifierStatusSuccess;
}

void Notifier::Close() {
  std::unique_lock<std::mutex> lock(mutex_);
  is_closed_ = true;
  cond_.notify_all();
}

}
