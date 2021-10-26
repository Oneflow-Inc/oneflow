/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_KERNEL_REENTRANT_LOCK_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_REENTRANT_LOCK_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/graph/graph.h"

namespace oneflow {

class ReentrantLockStatus final : public KernelState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReentrantLockStatus);
  ReentrantLockStatus() = default;
  ~ReentrantLockStatus() = default;

  void Init(const KernelConf& kernel_conf);

  static std::string kEmptyIbn;

  // true: success
  // false: failed
  void RequestLock(int64_t lock_id, std::queue<int64_t>* unlocked_ids);

  // return lock_id if any other lock acquired
  // -1: no other lock acquired
  void ReleaseLock(int64_t lock_id, std::queue<int64_t>* unlocked_ids);

  const std::queue<int64_t>& cur_unlocked_ids() const { return cur_unlocked_ids_; }
  std::queue<int64_t>* mut_cur_unlocked_ids() { return &cur_unlocked_ids_; }

  // Getters
  const std::string& cur_ibn() const { return cur_ibn_; }
  int64_t cur_act_id() const { return cur_act_id_; }
  bool acquired_lock_to_be_sent() const { return acquired_lock_to_be_sent_; }
  size_t total_queued_request_lock_num() const { return total_queued_request_lock_num_; }
  size_t total_acquired_lock_num() const { return total_acquired_lock_num_; }

  // Setters
  void set_cur_ibn(const std::string& ibn) { cur_ibn_ = ibn; }
  void set_cur_act_id(int64_t act_id) { cur_act_id_ = act_id; }
  void set_acquired_lock_to_be_sent(bool val) { acquired_lock_to_be_sent_ = val; }

 private:
  // true: success
  // false: failed
  bool TryAcquireLock(int64_t lock_id);

  std::string cur_ibn_;
  int64_t cur_act_id_{};
  bool acquired_lock_to_be_sent_{};
  size_t total_queued_request_lock_num_{};
  size_t total_acquired_lock_num_{};
  std::vector<std::queue<int64_t>> lock_id2queued_request_act_id_;
  std::vector<size_t> lock_id2acquired_num_;
  std::vector<std::vector<int64_t>> lock_id2intersecting_lock_ids_;
  std::queue<int64_t> cur_unlocked_ids_;
};

template<typename T>
class ReentrantLockKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReentrantLockKernel);
  ReentrantLockKernel() = default;
  ~ReentrantLockKernel() override = default;

 private:
  void VirtualKernelInit(KernelContext* ctx) override;
  void ForwardDataContent(KernelContext* ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REENTRANT_LOCK_KERNEL_H_
