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
#include "oneflow/core/kernel/reentrant_lock_kernel.h"

namespace oneflow {

std::string ReentrantLockStatus::kEmptyIbn = "reentrant_lock_status_empty_ibn";

void ReentrantLockStatus::Init(const KernelConf& kernel_conf) {
  const auto& conf = kernel_conf.op_attribute().op_conf().reentrant_lock_conf();
  cur_ibn_ = "";
  cur_act_id_ = -1;
  acquired_lock_to_be_sent_ = false;
  total_queued_request_lock_num_ = 0;
  total_acquired_lock_num_ = 0;
  lock_id2queued_request_act_id_.resize(conf.lock_id2intersecting_lock_ids_size());
  lock_id2acquired_num_.resize(conf.lock_id2intersecting_lock_ids_size());
  for (const Int64List& ids : conf.lock_id2intersecting_lock_ids()) {
    lock_id2intersecting_lock_ids_.emplace_back(
        std::vector<int64_t>(ids.value().begin(), ids.value().end()));
  }
}

bool ReentrantLockStatus::TryAcquireLock(int64_t lock_id) {
  CHECK_EQ(lock_id2queued_request_act_id_.at(lock_id).empty(), false);
  int64_t act_id = lock_id2queued_request_act_id_.at(lock_id).front();
  bool blocked = false;
  for (int64_t intersect_lock_id : lock_id2intersecting_lock_ids_.at(lock_id)) {
    if (lock_id2acquired_num_.at(intersect_lock_id) > 0
        || (lock_id2queued_request_act_id_.at(intersect_lock_id).empty() == false
            && lock_id2queued_request_act_id_.at(intersect_lock_id).front() < act_id)) {
      blocked = true;
      break;
    }
  }
  if (blocked) { return false; }
  lock_id2queued_request_act_id_.at(lock_id).pop();
  --total_queued_request_lock_num_;
  ++lock_id2acquired_num_.at(lock_id);
  ++total_acquired_lock_num_;
  return true;
}

void ReentrantLockStatus::RequestLock(int64_t lock_id, std::queue<int64_t>* unlocked_ids) {
  lock_id2queued_request_act_id_.at(lock_id).push(cur_act_id());
  ++total_queued_request_lock_num_;
  if (TryAcquireLock(lock_id)) { unlocked_ids->push(lock_id); }
}

void ReentrantLockStatus::ReleaseLock(int64_t lock_id, std::queue<int64_t>* unlocked_ids) {
  CHECK_GT(lock_id2acquired_num_.at(lock_id), 0);
  CHECK_GT(total_acquired_lock_num_, 0);
  --lock_id2acquired_num_.at(lock_id);
  --total_acquired_lock_num_;
  size_t unlocked_cnt = 0;
  do {
    unlocked_cnt = 0;
    auto ReleaseRelatedLockId = [&](int64_t related_lock_id) {
      if (lock_id2queued_request_act_id_.at(related_lock_id).empty()) { return; }
      if (TryAcquireLock(related_lock_id)) {
        unlocked_ids->push(related_lock_id);
        ++unlocked_cnt;
      }
    };
    ReleaseRelatedLockId(lock_id);
    for (int64_t id : lock_id2intersecting_lock_ids_.at(lock_id)) { ReleaseRelatedLockId(id); }
  } while (unlocked_cnt > 0);
}

template<typename T>
void ReentrantLockKernel<T>::VirtualKernelInit(KernelContext* ctx) {
  ctx->set_state(std::make_shared<ReentrantLockStatus>());
}

template<typename T>
void ReentrantLockKernel<T>::ForwardDataContent(KernelContext* ctx) const {
  auto* const status = CHECK_NOTNULL(dynamic_cast<ReentrantLockStatus*>(ctx->state().get()));
  if (status->cur_ibn() == "start") {
    T lock_id = *ctx->BnInOp2Blob("start")->dptr<T>();
    status->RequestLock(lock_id, status->mut_cur_unlocked_ids());
  } else if (status->cur_ibn() == "end") {
    status->ReleaseLock(*ctx->BnInOp2Blob("end")->dptr<T>(), status->mut_cur_unlocked_ids());
  } else {
    CHECK_EQ(status->cur_ibn(), ReentrantLockStatus::kEmptyIbn);
  }
  if (status->cur_unlocked_ids().size() > 0) {
    T lock_id = status->cur_unlocked_ids().front();
    status->mut_cur_unlocked_ids()->pop();
    *ctx->BnInOp2Blob("out")->mut_dptr<T>() = lock_id;
    status->set_acquired_lock_to_be_sent(true);
  } else {
    status->set_acquired_lock_to_be_sent(false);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kReentrantLockConf, ReentrantLockKernel,
                               INT_DATA_TYPE_SEQ)

}  // namespace oneflow
