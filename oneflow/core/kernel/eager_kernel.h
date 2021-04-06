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
#ifndef ONEFLOW_CORE_KERNEL_EAGER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_EAGER_KERNEL_H_

#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

class EagerKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerKernel);
  EagerKernel(const JobDesc* job_desc, const KernelConf& kernel_conf);
  ~EagerKernel() = default;

  void Infer(std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  std::shared_ptr<user_op::OpKernelState> EagerForward(
      const std::shared_ptr<user_op::OpKernelState>& old_opkernel_state, DeviceCtx* device_ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;

 private:
  void InitOpKernel(const KernelConf& kernel_conf);
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    UNIMPLEMENTED();
  }
  std::unique_ptr<const user_op::OpKernel> kernel_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_EAGER_KERNEL_H_
