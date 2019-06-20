#ifndef ONEFLOW_CORE_KERNEL_REENTRANT_LOCK_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_REENTRANT_LOCK_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/graph/graph.h"

namespace oneflow {

class ReentrantLockStatus final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReentrantLockStatus);
  ReentrantLockStatus() = default;
  ~ReentrantLockStatus() = default;

  void Init(const KernelConf& kernel_conf);

  // true: success
  // false: failed
  bool RequestLock(int64_t lock_id);

  // return lock_id if any other lock acquired
  // -1: no other lock acquired
  int64_t ReleaseLock(int64_t lock_id);

  // Getters
  const std::string& cur_ibn() const { return cur_ibn_; }
  int64_t cur_act_id() const { return cur_act_id_; }
  bool cur_act_one_lock_acquired() const { return cur_act_one_lock_acquired_; }
  size_t total_queued_request_lock_num() const { return total_queued_request_lock_num_; }
  size_t total_acquired_lock_num() const { return total_acquired_lock_num_; }

  // Setters
  void set_cur_ibn(const std::string& ibn) { cur_ibn_ = ibn; }
  void set_cur_act_id(int64_t act_id) { cur_act_id_ = act_id; }
  void set_cur_act_one_lock_acquired(bool val) { cur_act_one_lock_acquired_ = val; }

 private:
  // true: success
  // false: failed
  bool TryAcquireLock(int64_t lock_id);

  std::string cur_ibn_;
  int64_t cur_act_id_;
  bool cur_act_one_lock_acquired_;
  size_t total_queued_request_lock_num_;
  size_t total_acquired_lock_num_;
  std::vector<std::queue<int64_t>> lock_id2queued_request_act_id_;
  std::vector<size_t> lock_id2acquired_num_;
  std::vector<std::vector<int64_t>> lock_id2intersecting_lock_ids_;
};

template<typename T>
class ReentrantLockKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReentrantLockKernel);
  ReentrantLockKernel() = default;
  ~ReentrantLockKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REENTRANT_LOCK_KERNEL_H_
