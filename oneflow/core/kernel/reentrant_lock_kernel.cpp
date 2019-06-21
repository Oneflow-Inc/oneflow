#include "oneflow/core/kernel/reentrant_lock_kernel.h"

namespace oneflow {

void ReentrantLockStatus::Init(const KernelConf& kernel_conf) { TODO(); }

bool ReentrantLockStatus::RequestLock(int64_t lock_id) { TODO(); }

int64_t ReentrantLockStatus::ReleaseLock(int64_t lock_id) { TODO(); }

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
