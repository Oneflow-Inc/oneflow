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
#include "oneflow/core/kernel/foreign_watch_kernel.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/job/foreign_watcher.h"

namespace oneflow {

template<DeviceType device_type>
void ForeignWatchKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  OfBlob of_blob(ctx.device_ctx, BnInOp2Blob("in"));
  (*Global<std::shared_ptr<ForeignWatcher>>::Get())
      ->Call(this->op_conf().foreign_watch_conf().handler_uuid(),
             reinterpret_cast<int64_t>(&of_blob));
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kForeignWatchConf, DeviceType::kCPU,
                            ForeignWatchKernel<DeviceType::kCPU>);

#ifdef WITH_CUDA
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kForeignWatchConf, DeviceType::kGPU,
                            ForeignWatchKernel<DeviceType::kGPU>);
#endif

}  // namespace oneflow
