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
#ifndef ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class NormalMdUpdateKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalMdUpdateKernel);
  virtual ~NormalMdUpdateKernel() = default;

 protected:
  NormalMdUpdateKernel() = default;
  virtual void UpdateModel(DeviceCtx* ctx, T weight_decay, const int64_t* train_step,
                           const float* learning_rate,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual bool IsWeightDecaySupported() { return false; }

  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ForwardDataContent(ctx, BnInOp2Blob);
  }

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void VirtualKernelInit() override;

  T weight_decay_;
};

#define DECLARE_MDUPDT_KERNEL_CREATOR(x) Kernel* Create##x##MdUpdtKernel(const KernelConf&);

#define DEFINE_MDUPDT_KERNEL_CREATOR(x)                                                      \
  Kernel* Create##x##MdUpdtKernel(const KernelConf& kernel_conf) {                           \
    static const HashMap<std::string, std::function<Kernel*()>> creators = {                 \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY, (x##MdUpdateKernel),     \
                                         DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)};          \
    DeviceType device_type =                                                                 \
        CHECK_JUST(DeviceType4DeviceTag(kernel_conf.op_attribute().op_conf().device_tag())); \
    return creators.at(GetHashKey(device_type, kernel_conf.data_type()))();                  \
  }

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NORMAL_MODEL_UPDATE_KERNEL_H_
