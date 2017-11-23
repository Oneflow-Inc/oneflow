#ifndef ONEFLOW_CORE_COMMON_BLOCKING_COUNTER_H_
#define ONEFLOW_CORE_COMMON_BLOCKING_COUNTER_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class BlockingCounter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlockingCounter);
  BlockingCounter() = default;
  ~BlockingCounter() = default;

  void Init(const std::string& cnt_name, int64_t cnt_val) {
    cnt_name_ = cnt_name;
    cnt_val_ = cnt_val;
    LOG(INFO) << "Set " << cnt_name_ << " " << cnt_val_;
  }
  void Decrease() {
    std::unique_lock<std::mutex> lck(mtx_);
    cnt_val_ -= 1;
    LOG(INFO) << cnt_name_ << " minus one, current val is " << cnt_val_;
    if (cnt_val_ == 0) { cond_.notify_all(); }
  }
  void WaitUntilCntEqualZero() {
    std::unique_lock<std::mutex> lck(mtx_);
    cond_.wait(lck, [this]() { return cnt_val_ == 0; });
  }

 private:
  std::string cnt_name_;
  std::mutex mtx_;
  std::condition_variable cond_;
  int64_t cnt_val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_BLOCKING_COUNTER_H_
