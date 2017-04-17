#ifndef ONEFLOW_RUNTIME_MT_QUEUE_H_
#define ONEFLOW_RUNTIME_MT_QUEUE_H_
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
// Defines a concurrent queue.
namespace oneflow {
template <typename T>
class MtQueue {
  public:
    MtQueue() : exit_(false) {}
    ~MtQueue() {}

    void Push(const T& item);
    // If the queue is empty, the thread calling |Pop| would be blocked.
    // return true if successfully pop out an item, false when the queue tell
    // the owner thread should exit.
    bool Pop(T& item);

    // Return true if successfully pop out an item, false if the queue is empty.
    bool TryPop(T& item);

    // Get the number of items in the queue
    size_t Size() const;

    // Whether the queue is empty or not
    bool Empty() const;

    // Exit queue, awake all the threads blocked by the queue
    void Exit();

  private:
    std::queue<T> buffer_;
    mutable std::mutex mutex_;
    std::condition_variable empty_condition_;
    bool exit_;

    MtQueue(const MtQueue& other) = delete;
    MtQueue& operator=(const MtQueue& other) = delete;
};

template <typename T>
void MtQueue<T>::Push(const T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  buffer_.push(item);
  empty_condition_.notify_one();
}

template <typename T>
bool MtQueue<T>::Pop(T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  empty_condition_.wait(lock,
      [this] {
      return !buffer_.empty() || exit_;
      });
  if (buffer_.empty()) {
    return false;
  }
  item = buffer_.front();
  buffer_.pop();
  return true;
}

template <typename T>
bool MtQueue<T>::TryPop(T& item) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (buffer_.empty()) {
    return false;
  }
  item = buffer_.front();
  buffer_.pop();
  return true;
}

template <typename T>
size_t MtQueue<T>::Size() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return buffer_.size();
}

template <typename T>
bool MtQueue<T>::Empty() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return buffer_.empty();
}

template <typename T>
void MtQueue<T>::Exit() {
  std::unique_lock<std::mutex> lock(mutex_);
  exit_ = true;
  empty_condition_.notify_all();
}
}  // namespace oneflow
#endif  // ONEFLOW_RUNTIME_MT_QUEUE_H_
