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
#ifndef ONEFLOW_CORE_KERNEL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_H_

#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/kernel/kernel_registration.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

class JobDesc;
class RuntimeBlobShapeInferHelper;

class Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Kernel);
  virtual ~Kernel();

  void Init(const KernelConf& kernel_conf, KernelContext* ctx);
  void Launch(KernelContext* ctx) const;

  const OperatorConf& op_conf() const { return op_attribute().op_conf(); }
  const OpAttribute& op_attribute() const { return kernel_conf().op_attribute(); }
  const KernelConf& kernel_conf() const { return kernel_conf_; }
  /*
   * return true means all below must be guaranteed when `Launch` function return:
   * 1) all out blob header has been set (e.g. SyncSetHeadKernel)
   * 2) all asynchronous task has been queued (e.g. NCCL related kernel)
   */
  virtual bool IsKernelLaunchSynchronized() const { return true; }

  void SystemForwardHeader(KernelContext* ctx) const { ForwardHeader(ctx); }
  void SystemForwardDataContent(KernelContext* ctx) const { ForwardDataContent(ctx); }
  virtual void Forward(KernelContext* ctx) const;

 protected:
  Kernel();
  void InitBase(const KernelConf&);
  virtual void VirtualKernelInit(KernelContext* ctx) {}

  virtual void ForwardHeader(KernelContext* ctx) const;
  virtual void ForwardShape(KernelContext* ctx) const;
  // TODO(niuchong) : rename ForwardDataContent to ForwardBody
  virtual void ForwardDataContent(KernelContext* ctx) const = 0;
  virtual bool IsStateless() const { return false; }

 private:
  std::unique_ptr<RuntimeBlobShapeInferHelper> shape_infer_helper_;
  KernelConf kernel_conf_;
};

#define REGISTER_KERNEL(k, KernelType) \
  REGISTER_CLASS_WITH_ARGS(int32_t, k, Kernel, KernelType, const KernelConf&)
#define REGISTER_KERNEL_CREATOR(k, f) \
  REGISTER_CLASS_CREATOR(int32_t, k, Kernel, f, const KernelConf&)

std::unique_ptr<const Kernel> ConstructKernel(const KernelConf& kernel_conf, KernelContext* ctx);

}  // namespace oneflow

#define MAKE_KERNEL_CREATOR_ENTRY(kernel_class, device_type, data_type_pair) \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)),               \
   []() { return new kernel_class<device_type, OF_PP_PAIR_FIRST(data_type_pair)>(); }},

#define ADD_DEFAULT_KERNEL_CREATOR(op_type_case, kernel_class, data_type_seq)                \
  namespace {                                                                                \
                                                                                             \
  Kernel* OF_PP_CAT(CreateKernel, __LINE__)(const KernelConf& kernel_conf) {                 \
    static const HashMap<std::string, std::function<Kernel*()>> creators = {                 \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY, (kernel_class),          \
                                         DEVICE_TYPE_SEQ, data_type_seq)};                   \
    DeviceType device_type =                                                                 \
        CHECK_JUST(DeviceType4DeviceTag(kernel_conf.op_attribute().op_conf().device_tag())); \
    auto key = GetHashKey(device_type, kernel_conf.data_type());                             \
    auto it = creators.find(key);                                                            \
    if (it == creators.end()) {                                                              \
      LOG(FATAL) << "Error! Cannot find kernel creator: " << kernel_conf.DebugString()       \
                 << " with device_type = " << device_type                                    \
                 << ", dtype = " << kernel_conf.data_type();                                 \
    }                                                                                        \
    return (it->second)();                                                                   \
  }                                                                                          \
                                                                                             \
  REGISTER_KERNEL_CREATOR(op_type_case, OF_PP_CAT(CreateKernel, __LINE__));                  \
  }

#define MAKE_DEVICE_TYPE_KERNEL_CREATOR_ENTRY(kernel_class, device_type) \
  {device_type, []() { return new kernel_class<device_type>(); }},

#define ADD_DEVICE_TYPE_KERNEL_CREATOR(op_type_case, kernel_class)                              \
  namespace {                                                                                   \
                                                                                                \
  Kernel* OF_PP_CAT(CreateKernel, __LINE__)(const KernelConf& kernel_conf) {                    \
    static const HashMap<int, std::function<Kernel*()>> creators = {                            \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_DEVICE_TYPE_KERNEL_CREATOR_ENTRY, (kernel_class), \
                                         DEVICE_TYPE_SEQ)};                                     \
    DeviceType device_type =                                                                    \
        CHECK_JUST(DeviceType4DeviceTag(kernel_conf.op_attribute().op_conf().device_tag()));    \
    auto it = creators.find(device_type);                                                       \
    if (it == creators.end()) {                                                                 \
      LOG(FATAL) << "Error! Cannot find kernel creator: " << kernel_conf.DebugString()          \
                 << " with device_type = " << device_type;                                      \
    }                                                                                           \
    return (it->second)();                                                                      \
  }                                                                                             \
                                                                                                \
  REGISTER_KERNEL_CREATOR(op_type_case, OF_PP_CAT(CreateKernel, __LINE__));                     \
  }

#define MAKE_CPU_KERNEL_CREATOR_ENTRY(kernel_class, data_type_pair) \
  {OF_PP_PAIR_SECOND(data_type_pair),                               \
   []() { return new kernel_class<OF_PP_PAIR_FIRST(data_type_pair)>(); }},

#define ADD_CPU_DEFAULT_KERNEL_CREATOR(op_type_case, kernel_class, data_type_seq)       \
  namespace {                                                                           \
                                                                                        \
  Kernel* CreateKernel(const KernelConf& kernel_conf) {                                 \
    static const HashMap<int, std::function<Kernel*()>> creators = {                    \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_CPU_KERNEL_CREATOR_ENTRY, (kernel_class), \
                                         data_type_seq)};                               \
    auto it = creators.find(kernel_conf.data_type());                                   \
    if (it == creators.end()) {                                                         \
      LOG(FATAL) << "Error! Cannot find kernel creator: " << kernel_conf.DebugString()  \
                 << " with dtype = " << kernel_conf.data_type();                        \
    }                                                                                   \
    return (it->second)();                                                              \
  }                                                                                     \
                                                                                        \
  REGISTER_KERNEL_CREATOR(op_type_case, CreateKernel);                                  \
  }

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_H_
