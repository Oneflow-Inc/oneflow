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
#include "oneflow/core/kernel/output_kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

template<DeviceType device_type>
void OutputKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  LOG(ERROR) << "OutputKernel, in shape: " << BnInOp2Blob("in")->shape();
  if (CHECK_JUST(*Global<Maybe<bool>, MultiClient>::Get())) {
    const auto& job_name = this->job_desc().job_name();
    const auto& op_name = this->op_conf().name();
    auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
    auto* buffer = buffer_mgr->Get(GetOutputBufferName(job_name, op_name));
    std::shared_ptr<JobInstance> job_instance;
    BufferStatus buffer_status = buffer->TryReceive(&job_instance);
    LOG(ERROR) << "OutputKernel, buffer received, job_name: " << job_name
               << ", op_name: " << op_name << ", status: " << buffer_status;
    CHECK_NE(buffer_status, kBufferStatusEmpty);
    if (buffer_status == kBufferStatusSuccess) {
      Blob* in_blob = BnInOp2Blob("in");
      OfBlob ofblob(ctx.device_ctx, in_blob);
      std::ostringstream ss;
      ss << "OutputKernel in_blob, shape: " << in_blob->shape().ToString();
      ss << ", data: [";
      for (int i = 0; i < 5; ++i) {
        ss << in_blob->dptr<float>()[i];
        if (i != 4) {
          ss << ", ";
        }
      }
      ss << "]";
      LOG(ERROR) << ss.str();
      LOG(ERROR) << "OutputKernel, PullBlobByOpName, op_name: " << op_name;
      job_instance->PullBlobByOpName(reinterpret_cast<uint64_t>(&ofblob), op_name);
    }
  } else {
    BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
  }
}

template<DeviceType device_type>
void OutputKernel<device_type>::ForwardHeader(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (CHECK_JUST(*Global<Maybe<bool>, MultiClient>::Get())) {
    // Do nothing.
  } else {
    BnInOp2Blob("out")->CopyHeaderFrom(ctx.device_ctx, BnInOp2Blob("in"));
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kOutputConf, OutputKernel);

}  // namespace oneflow
