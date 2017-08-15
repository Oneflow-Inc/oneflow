#ifndef ONEFLOW_CORE_COMMON_CHANNEL_H_
#define ONEFLOW_CORE_COMMON_CHANNEL_H_

#include "oneflow/core/common/concurrent_queue.h"
#include "oneflow/core/common/util.h"

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

  // close the channel's receive end , the thread can't receive item from
  // channel
  void CloseReceiveEnd();

 private:
  moodycamel::ConcurrentQueue<T> val_;
  std::atomic<bool> is_send_closed_;
  std::atomic<int64_t> item_cnt_;
  std::atomic<bool> is_receive_closed_;
};

template<typename T>
int Channel<T>::Send(const T& item) {
  if (of_unlikely(is_send_closed_)) { return -1; }
  item_cnt_ += val_.enqueue(item);

  return 0;
}

template<typename T>
int Channel<T>::Receive(T* item) {
  if (of_unlikely(is_receive_closed_ || (is_send_closed_ && item_cnt_ == 0))) {
    return -1;
  }
  return !val_.try_dequeue(*item);
}

template<typename T>
void Channel<T>::CloseSendEnd() {
  is_send_closed_ = true;
}

template<typename T>
void Channel<T>::CloseReceiveEnd() {
  is_receive_closed_ = true;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_CHANNEL_H_
