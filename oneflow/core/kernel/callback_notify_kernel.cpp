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
class CallbackNotifyKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CallbackNotifyKernel);
  CallbackNotifyKernel() = default;
  ~CallbackNotifyKernel() = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(KernelContext* ctx) const override;
};

template<typename T>
void CallbackNotifyKernel<T>::ForwardDataContent(KernelContext* ctx) const {
  auto* buffer_mgr = Singleton<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
  std::string buffer_name;
  CHECK(this->op_conf().callback_notify_conf().has_job_name());
  buffer_name = GetCallbackNotifierBufferName(this->op_conf().callback_notify_conf().job_name());
  std::shared_ptr<JobInstance> job_instance;
  BufferStatus buffer_status = buffer_mgr->Get(buffer_name)->TryReceive(&job_instance);
  CHECK_NE(buffer_status, kBufferStatusEmpty);
  if (buffer_status == kBufferStatusSuccess) { job_instance->Finish(); }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCallbackNotifyConf, CallbackNotifyKernel,
                               INT_DATA_TYPE_SEQ);

}  // namespace oneflow
