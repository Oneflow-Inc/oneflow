#ifndef ONEFLOW_CORE_COMMON_PRIOR_MUTEX_H_
#define ONEFLOW_CORE_COMMON_PRIOR_MUTEX_H_

#include <mutex>

namespace oneflow {

template <typename T>
class HighPriorUniqueLock;

template <typename T>
class LowPriorUniqueLock;

class PriorMutex final {
 public:
  PriorMutex() : mutex_(), is_hurry_(false) {}
  ~PriorMutex() = default;

  PriorMutex(const PriorMutex&) = delete;
  PriorMutex(PriorMutex&&) = delete;
  PriorMutex& operator=(const PriorMutex&) = delete;
  PriorMutex& operator=(PriorMutex&&) = delete;

 private:
  friend class HighPriorUniqueLock<PriorMutex>;
  friend class LowPriorUniqueLock<PriorMutex>;

  std::mutex* mut_mutex() { return &mutex_; }
  volatile bool* mut_is_hurry() { return &is_hurry_; }

  std::mutex mutex_;
  volatile bool is_hurry_;
};

template<>
class LowPriorUniqueLock<PriorMutex> final {
 public:
  explicit LowPriorUniqueLock(PriorMutex* prior_mutex) : prior_mutex_(prior_mutex) {
    prior_mutex->mut_mutex()->lock();
  }
  ~LowPriorUniqueLock() {
    prior_mutex->mut_mutex()->unlock();
  }
  LowPriorUniqueLock(const LowPriorUniqueLock&) = delete;
  LowPriorUniqueLock(LowPriorUniqueLock&&) = delete;
  LowPriorUniqueLock& operator=(const LowPriorUniqueLock&) = delete;
  LowPriorUniqueLock& operator=(LowPriorUniqueLock&&) = delete;

  bool TestIsHurryAndClearHurry() {
    bool is_hurry = *prior_mutex_->mut_is_hurry();
    if (is_hurry) { *prior_mutex_->mut_is_hurry() = false; }
    return is_hurry;
  }

 private:
  PriorMutex* prior_mutex_;
  std::unique_lock<std::mutex> lock_;
};

template<>
class HighPriorUniqueLock<PriorMutex> final {
 public:
  explicit HighPriorUniqueLock(PriorMutex* prior_mutex) : prior_mutex_(prior_mutex) {
    prior_mutex->mut_is_hurry() = true;
    while (!prior_mutex->mut_mutex()->try_lock()) {}
  }
  ~HighPriorUniqueLock() {
    prior_mutex->mut_mutex()->unlock();
    prior_mutex->mut_is_hurry() = false;
  }
  HighPriorUniqueLock(const HighPriorUniqueLock&) = delete;
  HighPriorUniqueLock(HighPriorUniqueLock&&) = delete;
  HighPriorUniqueLock& operator=(const HighPriorUniqueLock&) = delete;
  HighPriorUniqueLock& operator=(HighPriorUniqueLock&&) = delete;

 private:
  PriorMutex* prior_mutex_;
};

}

#endif  // ONEFLOW_CORE_COMMON_PRIOR_MUTEX_H_
