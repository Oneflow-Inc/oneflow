#include "oneflow/core/kernel/reentrant_lock_kernel.h"

namespace oneflow {

void ReentrantLockStatus::Init(const KernelConf& kernel_conf) {
  const auto& conf = kernel_conf.op_attribute().op_conf().reentrant_lock_conf();
  cur_ibn_ = "";
  cur_act_id_ = -1;
  cur_act_one_lock_acquired_ = false;
  total_queued_request_lock_num_ = 0;
  total_acquired_lock_num_ = 0;
  lock_id2queued_request_act_id_.resize(conf.lock_id2intersecting_lock_ids_size());
  lock_id2acquired_num_.resize(conf.lock_id2intersecting_lock_ids_size());
  lock_id2intersecting_lock_ids_ = conf.lock_id2intersecting_lock_ids();
}

bool ReentrantLockStatus::TryAcquireLock(int64_t lock_id) {
  CHECK_EQ(lock_id2queued_request_act_id_.at(lock_id).empty(), false);
  int64_t act_id = lock_id2queued_request_act_id_.at(lock_id).front();
  bool blocked = false;
  for (int64_t intersect_lock_id : lock_id2intersecting_lock_ids_.Get(lock_id).id()) {
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

bool ReentrantLockStatus::RequestLock(int64_t lock_id) {
  lock_id2queued_request_act_id_.at(lock_id).push(cur_act_id());
  ++total_queued_request_lock_num_;
  return TryAcquireLock(lock_id);
}

int64_t ReentrantLockStatus::ReleaseLock(int64_t lock_id) {
  CHECK_GT(lock_id2acquired_num_.at(lock_id), 0);
  CHECK_GT(total_acquired_lock_num_, 0);
  --lock_id2acquired_num_.at(lock_id);
  --total_acquired_lock_num_;
  if (lock_id2acquired_num_.at(lock_id) == 0) {
    int64_t min_act_id = cur_act_id();
    int64_t min_lock_id = -1;
    for (int64_t intersect_lock_id : lock_id2intersecting_lock_ids_.Get(lock_id).id()) {
      CHECK_EQ(lock_id2acquired_num_.at(intersect_lock_id), 0);
      int64_t act_id = lock_id2queued_request_act_id_.at(intersect_lock_id).front();
      if (act_id < min_act_id) {
        min_act_id = act_id;
        min_lock_id = intersect_lock_id;
      }
    }
    if (min_lock_id != -1 && TryAcquireLock(min_lock_id)) { return min_lock_id; }
  }
  return -1;
}

template<typename T>
void ReentrantLockKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* status = static_cast<ReentrantLockStatus*>(ctx.other);
  T lock_id = -1;
  if (status->cur_ibn() == "start") {
    lock_id = *BnInOp2Blob("start")->dptr<T>();
    status->set_cur_act_one_lock_acquired(status->RequestLock(lock_id));
  } else if (status->cur_ibn() == "end") {
    lock_id = status->ReleaseLock(*BnInOp2Blob("end")->dptr<T>());
    status->set_cur_act_one_lock_acquired(lock_id != -1);
  } else {
    UNIMPLEMENTED();
  }
  if (status->cur_act_one_lock_acquired()) { *BnInOp2Blob("out")->mut_dptr<T>() = lock_id; }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kReentrantLockConf, ReentrantLockKernel,
                               INT_DATA_TYPE_SEQ)

}  // namespace oneflow
