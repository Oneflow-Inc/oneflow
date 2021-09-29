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

class CriticalSectionEndTickKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CriticalSectionEndTickKernel);
  CriticalSectionEndTickKernel() = default;
  ~CriticalSectionEndTickKernel() = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(KernelContext* ctx) const override;
};

void CriticalSectionEndTickKernel::ForwardDataContent(KernelContext* ctx) const {
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<CriticalSectionInstance>>>::Get();
  bool is_multi_client = CHECK_JUST(*Global<Maybe<bool>, MultiClient>::Get());
  CHECK(op_conf().has_critical_section_end_conf());
  std::string buffer_name = GetCriticalSectionEndBufferName(op_conf().critical_section_end_conf().job_name());
  std::shared_ptr<CriticalSectionInstance> foreign_critical_section_instance;
  BufferStatus buffer_status = buffer_mgr->Get(buffer_name)->TryReceive(&foreign_critical_section_instance);
  CHECK_EQ(buffer_status, kBufferStatusSuccess);
}

REGISTER_KERNEL(OperatorConf::kCriticalSectionEndTickConf, CriticalSectionEndTickKernel);

}  // namespace oneflow
