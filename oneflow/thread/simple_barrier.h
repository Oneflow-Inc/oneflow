#ifndef _SIMPLE_BARRIER_H_
#define _SIMPLE_BARRIER_H_
#include <thread>
#include <atomic>
#include <condition_variable>

namespace caffe {
class SimpleBarrier {
  public:
    SimpleBarrier(int32_t num_threads)
     : num_in_waiting_(0),barrier_size_(num_threads) {}
    ~SimpleBarrier() {}
    bool Wait();

  private:
    int32_t barrier_size_;
    int32_t num_in_waiting_;
    std::condition_variable cond_;
    std::mutex mutex_;

    SimpleBarrier(const SimpleBarrier& other) = delete;
    SimpleBarrier& operator=(const SimpleBarrier& other) = delete;
};

inline bool SimpleBarrier::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  if ((++num_in_waiting_) == barrier_size_) {
    num_in_waiting_ = 0;
    cond_.notify_all();
    return true;
  } else {
    cond_.wait(lock);
    return false;
  }
}
}  // namespace caffe
#endif  // _SIMPLE_BARRIER_H_
