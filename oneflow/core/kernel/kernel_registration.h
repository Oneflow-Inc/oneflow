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
#ifndef ONEFLOW_CORE_KERNEL_KERNEL_REGISTRATION_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_REGISTRATION_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/operator/op_conf_util.h"

namespace oneflow {

class Kernel;

namespace kernel_registration {

using CreateFn = std::function<Kernel*()>;
using IsMatchedPredicator = std::function<bool(const KernelConf&)>;

class KernelConstraint final {
 public:
  KernelConstraint() = default;
  ~KernelConstraint() = default;

  bool IsMatched(const KernelConf& conf) const { return predicator_(conf); }
  void SetIsMatchedPred(IsMatchedPredicator pred) { predicator_ = pred; }

 private:
  IsMatchedPredicator predicator_;
};

struct KernelRegistryVal final {
  KernelRegistryVal() : func(), cons() {}

  CreateFn func;
  KernelConstraint cons;
};

class KernelRegistrarBuilder final {
 public:
  explicit KernelRegistrarBuilder(OperatorConf::OpTypeCase op_type)
      : op_type_(op_type), registry_val_() {}
  KernelRegistrarBuilder& SetCreateFn(CreateFn fn);
  KernelRegistrarBuilder& SetIsMatchedPred(IsMatchedPredicator fn);

  void Finalize(OperatorConf::OpTypeCase* op_type, KernelRegistryVal* val) const;

 private:
  OperatorConf::OpTypeCase op_type_;
  KernelRegistryVal registry_val_;
};

struct KernelRegistrar final {
  KernelRegistrar(const KernelRegistrarBuilder&);
};

Kernel* CreateKernel(const KernelConf& kernel_conf);

}  // namespace kernel_registration

#define NEW_REGISTER_KERNEL(op_type, ...)                                           \
  static kernel_registration::KernelRegistrar OF_PP_CAT(g_registrar, __COUNTER__) = \
      kernel_registration::KernelRegistrarBuilder(op_type).SetCreateFn(             \
          []() { return new __VA_ARGS__(); })

#define REGISTER_KERNEL_WITH_NOTHING(op_type, ...)                                           \
  NEW_REGISTER_KERNEL(op_type, __VA_ARGS__).SetIsMatchedPred([](const KernelConf&) -> bool { \
    return true;                                                                             \
  });

#define REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, device, dtype, ...)                        \
  NEW_REGISTER_KERNEL(op_type, __VA_ARGS__).SetIsMatchedPred([](const KernelConf& conf) -> bool { \
    return (ToString(device) == conf.op_attribute().op_conf().device_tag())                       \
           && (GetDataType<dtype>::value == conf.data_type());                                    \
  });

#define REGISTER_KERNEL_WITH_DEVICE(op_type, device, ...)                                         \
  NEW_REGISTER_KERNEL(op_type, __VA_ARGS__).SetIsMatchedPred([](const KernelConf& conf) -> bool { \
    return (ToString(device) == conf.op_attribute().op_conf().device_tag());                      \
  });

#define REGISTER_KERNEL_HELPER_CPU_FLOATING(op_type, kernel)               \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, DeviceType::kCPU, float,  \
                                        kernel<DeviceType::kCPU, float>)   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, DeviceType::kCPU, double, \
                                        kernel<DeviceType::kCPU, double>)

#define REGISTER_KERNEL_HELPER_GPU_FLOATING(op_type, kernel)               \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, DeviceType::kGPU, float,  \
                                        kernel<DeviceType::kGPU, float>)   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, DeviceType::kGPU, double, \
                                        kernel<DeviceType::kGPU, double>)

#define REGISTER_KERNEL_HELPER_GPU_HALF(op_type, kernel)                    \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, DeviceType::kGPU, float16, \
                                        kernel<DeviceType::kGPU, float16>)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_REGISTRATION_H_
