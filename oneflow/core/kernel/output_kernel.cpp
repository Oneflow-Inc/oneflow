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

namespace oneflow {

template<DeviceType device_type>
class OutputKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OutputKernel);
  OutputKernel() = default;
  ~OutputKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx& ctx,
      const std::function<Blob*(const std::string&)>& BnInOp2Blob) const override;
  void ForwardHeader(const KernelCtx& ctx,
                     const std::function<Blob*(const std::string&)>& BnInOp2Blob) const override;
};

template<DeviceType device_type>
void OutputKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  if (CHECK_JUST(*Global<Maybe<bool>, MultiClient>::Get())) {
    const auto& job_name = this->job_desc().job_name();
    const auto& op_name = this->op_conf().name();
    auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
    auto* buffer = buffer_mgr->Get(GetOutputBufferName(job_name, op_name));
    std::shared_ptr<JobInstance> job_instance;
    BufferStatus buffer_status = buffer->TryReceive(&job_instance);
    CHECK_NE(buffer_status, kBufferStatusEmpty);
    if (buffer_status == kBufferStatusSuccess) {
      OfBlob ofblob(ctx.device_ctx, BnInOp2Blob("in"));
      job_instance->PullBlobByOpName(reinterpret_cast<uint64_t>(&ofblob), op_name);
    }
  } else {
    BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
  }
}

template<DeviceType device_type>
void OutputKernel<device_type>::ForwardHeader(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  if (CHECK_JUST(*Global<Maybe<bool>, MultiClient>::Get())) {
    // Do nothing.
  } else {
    BnInOp2Blob("out")->CopyHeaderFrom(ctx.device_ctx, BnInOp2Blob("in"));
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kOutputConf, OutputKernel);

}  // namespace oneflow
