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

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kCallbackNotifyConf, DeviceType::kCPU, int8_t,
                                      CallbackNotifyKernel<int8_t>);
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kCallbackNotifyConf, DeviceType::kCPU, int32_t,
                                      CallbackNotifyKernel<int32_t>);
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kCallbackNotifyConf, DeviceType::kCPU, int64_t,
                                      CallbackNotifyKernel<int64_t>);

}  // namespace oneflow
