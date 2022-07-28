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
#include "oneflow/core/job/critical_section_instance.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/buffer_manager.h"

namespace oneflow {

class CriticalSectionCallbackTickKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CriticalSectionCallbackTickKernel);
  CriticalSectionCallbackTickKernel() = default;
  ~CriticalSectionCallbackTickKernel() = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(KernelContext* ctx) const override;
};

void CriticalSectionCallbackTickKernel::ForwardDataContent(KernelContext* ctx) const {
  auto* buffer_mgr = Singleton<BufferMgr<std::shared_ptr<CriticalSectionInstance>>>::Get();
  CHECK(op_conf().has_critical_section_callback_tick_conf());
  const std::string& buffer_name = op_conf().critical_section_callback_tick_conf().buffer_name();
  std::shared_ptr<CriticalSectionInstance> critical_section_instance;
  BufferStatus buffer_status = buffer_mgr->Get(buffer_name)->TryReceive(&critical_section_instance);
  CHECK_EQ(buffer_status, kBufferStatusSuccess);
  critical_section_instance->Finish();
}

REGISTER_KERNEL(OperatorConf::kCriticalSectionCallbackTickConf, CriticalSectionCallbackTickKernel);

}  // namespace oneflow
