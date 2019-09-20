#include "oneflow/core/kernel/callback_notify_kernel.h"
#include "oneflow/core/job/foreign_job_instance.h"

namespace oneflow {

template<typename T>
void CallbackNotifyKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  T job_id = *BnInOp2Blob("in")->dptr<T>();
  const auto& buffer_name = this->op_conf().callback_notify_conf().callback_buffer_name(job_id);
  std::shared_ptr<ForeignJobInstance> foreign_job_instance;
  BufferStatus buffer_status = Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Get()
                                   ->Get(buffer_name)
                                   ->TryReceive(&foreign_job_instance);
  CHECK_NE(buffer_status, kBufferStatusEmpty);
  if (buffer_status == kBufferStatusSuccess) { foreign_job_instance->Finish(); }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCallbackNotifyConf, CallbackNotifyKernel,
                               INT_DATA_TYPE_SEQ);

}  // namespace oneflow
