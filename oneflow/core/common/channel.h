#ifndef ONEFLOW_CORE_COMMON_CHANNEL_H_
#define ONEFLOW_CORE_COMMON_CHANNEL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

enum ChannelStatus { kChannelStatusSuccess = 0, kChannelStatusErrorClosed };

template<typename T>
class Channel final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Channel);
  Channel() : is_closed_(false) {}
  ~Channel() = default;

  ChannelStatus Send(const T& item);
  ChannelStatus Receive(T* item);
  ChannelStatus ReceiveMany(std::queue<T>* items);
  void Close();

 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  bool is_closed_;
  std::condition_variable cond_;
};

template<typename T>
ChannelStatus Channel<T>::Send(const T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (is_closed_) { return kChannelStatusErrorClosed; }
  queue_.push(item);
  cond_.notify_one();
  return kChannelStatusSuccess;
}

template<typename T>
ChannelStatus Channel<T>::Receive(T* item) {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this]() { return (!queue_.empty()) || is_closed_; });
  if (queue_.empty()) { return kChannelStatusErrorClosed; }
  *item = queue_.front();
  queue_.pop();
  return kChannelStatusSuccess;
}

template<typename T>
ChannelStatus Channel<T>::ReceiveMany(std::queue<T>* items) {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this]() { return (!queue_.empty()) || is_closed_; });
  if (queue_.empty()) { return kChannelStatusErrorClosed; }
  while (!queue_.empty()) {
    items->push(std::move(queue_.front()));
    queue_.pop();
  }
  return kChannelStatusSuccess;
}

template<typename T>
void Channel<T>::Close() {
  std::unique_lock<std::mutex> lock(mutex_);
  is_closed_ = true;
  cond_.notify_all();
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CHANNEL_H_
