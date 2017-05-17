#ifndef ONEFLOW_COMMON_CHANNEL_H_
#define ONEFLOW_COMMON_CHANNEL_H_

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <common/util.h>

namespace oneflow {

template<typename T>
class Channel {
public:
  OF_DISALLOW_COPY_AND_MOVE(Channel);
  Channel() : is_send_closed_(false), is_receive_closed_(false) {}
  ~Channel() = default;

  void Send(const T& item);

  //  If the channel is empty, the thread calling Receive() would be blocked.
  //  return value
  //    true: if successfully get the item ref in val_
  //    false: when the channel tell the owner thread should exit
  bool Receive(T& item);

  // close the channel's send end, the thread can't send item to the channel
  void CloseSendEnd() {
    std::unique_lock<std::mutex> lock(mutex_);
    is_send_closed_ = true;
  }

  // close the channel's receive end , the thread can't receive item from channel
  void CloseReceiveEnd() {
    std::unique_lock<std::mutex> lock(mutex_);
    is_receive_closed_ = true;
    cond_.notify_all();
  };

private:
  std::queue<T> val_;
  mutable std::mutex mutex_;
  bool is_send_closed_;
  bool is_receive_closed_;
  std::condition_variable cond_;
};

template<typename T>
void Channel<T>::Send(const T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (is_send_closed_) {
    return;
  }
  val_.push(item);
  cond_.notify_one();
}

template<typename T>
bool Channel<T>::Receive(T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (is_receive_closed_) {
    return false;
  }
  cond_.wait(lock, [this]() { return !val_.empty() || is_receive_closed_; });
  if (val_.empty()) {
    return false;
  }
  item = val_.front();
  val_.pop();
  return true;
}

}  // namespace oneflow

#endif  // ONEFLOW_COMMON_CHANNEL_H_
