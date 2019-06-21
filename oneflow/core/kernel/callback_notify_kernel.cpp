#include "oneflow/core/kernel/callback_notify_kernel.h"

namespace oneflow {

template<typename T>
void CallbackNotifyKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  T buffer_id = *BnInOp2Blob("in")->dptr<T>();
  const auto& buffer_name = this->op_conf().callback_notify_conf().callback_buffer_name(buffer_id);
  std::function<void()> Callback;
  BufferStatus buffer_status =
      Global<BufferMgr<std::function<void()>>>::Get()->Get(buffer_name)->TryReceive(&Callback);
  CHECK_NE(buffer_status, kBufferStatusEmpty);
  if (buffer_status == kBufferStatusSuccess) { Callback(); }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCallbackNotifyConf, CallbackNotifyKernel,
                               INT_DATA_TYPE_SEQ);

}  // namespace oneflow
