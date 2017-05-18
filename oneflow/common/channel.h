#ifndef ONEFLOW_COMMON_CHANNEL_H_
#define ONEFLOW_COMMON_CHANNEL_H_

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include "glog/logging.h"
#include "common/util.h"

namespace oneflow {

template<typename T>
class Channel final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Channel);
  Channel() : is_send_closed_(false), is_receive_closed_(false) {}
  ~Channel() = default;

  // return code
  //   0 : success send item
  //  -1 : fail (send end has been closed)
  int Send(const T& item);

  //  If the channel is empty, the thread calling Receive() would be blocked.
  //  return value
  //    0: success -- if successfully get the item ref in val_
  //    -1: fail -- when the channel tell the owner thread should exit
  int Receive(T* item);

  // close the channel's send end, the thread can't send item to the channel
  void CloseSendEnd();

  // close the channel's receive end , the thread can't receive item from channel
  void CloseReceiveEnd();

 private:
  std::queue<T> val_;
  mutable std::mutex mutex_;
  bool is_send_closed_;
  bool is_receive_closed_;
  std::condition_variable cond_;
};

template<typename T>
int Channel<T>::Send(const T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (is_send_closed_) {
    return -1;
  }
  val_.push(item);
  cond_.notify_one();
  return 0;
}

template<typename T>
int Channel<T>::Receive(T* item) {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this]() { return !val_.empty() || is_receive_closed_ || is_send_closed_; });
  if (val_.empty() || is_receive_closed_) {
    return -1;
  }
  *item = val_.front();
  val_.pop();
  return 0;
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

#endif  // ONEFLOW_COMMON_CHANNEL_H_
