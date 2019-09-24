#ifndef ONEFLOW_CORE_DATA_RING_QUEUE_H_
#define ONEFLOW_CORE_DATA_RING_QUEUE_H_

#include <thread>

namespace oneflow {
namespace util {

template<typename T>
class RingQueue final {
 public:
  RingQueue(size_t qsize) : queue_(qsize), head_(0), tail_(0) {}

  void Enqueue(std::unique_ptr<T>&& item);
  void SyncEnqueue(std::unique_ptr<T>&& item);
  std::unique_ptr<T> Dequeue();
  std::unique_ptr<T> SyncDequeue();
  std::unique_ptr<T> SyncDequeue(std::function<bool(const T*)> pred);
  bool IsFull() { return head_ == tail_ && queue_.at(head_); }
  bool IsEmpty() { return head_ == tail_ && !queue_.at(head_); }
  T* Head() const { return queue_.at(head_).get(); }
  T* Tail() const { return queue_.at(tail_).get(); }
  void Close() { is_closed_ = true; }

 private:
  std::vector<std::unique_ptr<T>> queue_;
  size_t head_;
  size_t tail_;
  bool is_closed_;
  std::mutex mtx_;
  std::condition_variable full_cv_;
  std::condition_variable empty_cv_;
};

template<typename T>
void RingQueue<T>::Enqueue(std::unique_ptr<T>&& item) {
  CHECK(!IsFull());
  CHECK(!queue_.at(tail_));
  queue_.at(tail_).swap(item);
  tail_ += 1;
  tail_ %= queue_.size();
  empty_cv_.notify_one();
}

template<typename T>
void RingQueue<T>::SyncEnqueue(std::unique_ptr<T>&& item) {
  std::unique_lock<std::mutex> lck(mtx_);
  full_cv_.wait(lck, [this] { return !IsFull() || is_closed_; });
  if (is_closed_) { return; }
  Enqueue(std::forward<std::unique_ptr<T>>(item));
}

template<typename T>
std::unique_ptr<T> RingQueue<T>::Dequeue() {
  CHECK(!IsEmpty());
  CHECK(queue_.at(head_));
  std::unique_ptr<T> ret(nullptr);
  ret.swap(queue_.at(head_));
  head_ += 1;
  head_ %= queue_.size();
  full_cv_.notify_one();
  return ret;
}

template<typename T>
std::unique_ptr<T> RingQueue<T>::SyncDequeue() {
  std::unique_lock<std::mutex> lck(mtx_);
  empty_cv_.wait(lck, [this] { return !IsEmpty() || is_closed_; });
  if (is_closed_) { return std::unique_ptr<T>(); }
  return Dequeue();
}

template<typename T>
std::unique_ptr<T> RingQueue<T>::SyncDequeue(std::function<bool(const T*)> pred) {
  std::unique_lock<std::mutex> lck(mtx_);
  empty_cv_.wait(lck, [this, pred] { return (!IsEmpty() && pred(Head())) || is_closed_; });
  if (is_closed_) { return std::unique_ptr<T>(); }
  return Dequeue();
}

}  // namespace util
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DATA_RING_QUEUE_H_
