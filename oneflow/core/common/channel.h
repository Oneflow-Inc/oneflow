#ifndef ONEFLOW_CORE_COMMON_CHANNEL_H_
#define ONEFLOW_CORE_COMMON_CHANNEL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

enum ChannelStatus { kChannelStatusSuccess = 0, kChannelStatusErrorClosed };

template<typename T>
class Channel final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Channel);
  Channel() : is_send_closed_(false), is_receive_closed_(false) {}
  ~Channel() = default;

  ChannelStatus Send(const T& item);
  ChannelStatus Receive(T *item);
  void CloseSendEnd();
  void CloseReceiveEnd();

 private:
  std::queue<T> val_;
  mutable std::mutex mutex_;
  bool is_send_closed_;
  bool is_receive_closed_;
  std::condition_variable cond_;
};

template<typename T>
ChannelStatus Channel<T>::Send(const T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (is_send_closed_) { return kChannelStatusErrorClosed; }
  val_.push(item);
  cond_.notify_one();
  return kChannelStatusSuccess;
}

template<typename T>
ChannelStatus Channel<T>::Receive(T* item) {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this]() { return !val_.empty() || is_receive_closed_ || is_send_closed_; });
  if (val_.empty() || is_receive_closed_) { return kChannelStatusErrorClosed; }
  *item = val_.front();
  val_.pop();
  return kChannelStatusSuccess;
}

template<typename T>
void Channel<T>::CloseSendEnd() {
  std::unique_lock<std::mutex> lock(mutex_);
  is_send_closed_ = true;
  cond_.notify_all();
}

template<typename T>
void Channel<T>::CloseReceiveEnd() {
  std::unique_lock<std::mutex> lock(mutex_);
  is_receive_closed_ = true;
  cond_.notify_all();
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CHANNEL_H_
