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
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/buffer_manager.h"

namespace oneflow {

template<typename T>
class CallbackNotifyKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CallbackNotifyKernel);
  CallbackNotifyKernel() = default;
  ~CallbackNotifyKernel() = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(
      const KernelCtx& ctx,
      const std::function<Blob*(const std::string&)>& BnInOp2Blob) const override;
};

template<typename T>
void CallbackNotifyKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
  std::string buffer_name;
  if (CHECK_JUST(*Global<Maybe<bool>, MultiClient>::Get())) {
    buffer_name = GetCallbackNotifierBufferName(this->job_desc().job_name());
  } else {
    T job_id = *BnInOp2Blob("in")->dptr<T>();
    buffer_name = this->op_conf().callback_notify_conf().callback_buffer_name(job_id);
  }
  std::shared_ptr<JobInstance> foreign_job_instance;
  BufferStatus buffer_status = buffer_mgr->Get(buffer_name)->TryReceive(&foreign_job_instance);
  CHECK_NE(buffer_status, kBufferStatusEmpty);
  if (buffer_status == kBufferStatusSuccess) { foreign_job_instance->Finish(); }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCallbackNotifyConf, CallbackNotifyKernel,
                               INT_DATA_TYPE_SEQ);

}  // namespace oneflow
