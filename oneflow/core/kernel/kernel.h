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

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/kernel/kernel_registration.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/op_conf_util.h"
#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

class RuntimeBlobShapeInferHelper;

class Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Kernel);
  virtual ~Kernel();

  const JobDesc& job_desc() const { return *job_desc_; }

  void Init(const JobDesc* job_desc, const KernelConf&, DeviceCtx*);

  void InitModelAndConstBuf(const KernelCtx& ctx,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  void Launch(const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  const LogicalBlobId& BnInOp2Lbi(const std::string& bn_in_op) const;
  const OperatorConf& op_conf() const { return op_attribute().op_conf(); }
  const OpAttribute& op_attribute() const { return kernel_conf().op_attribute(); }
  /*
   * return true means all below must be guaranteed when `Launch` function return:
   * 1) all out blob header has been set (e.g. SyncSetHeadKernel)
   * 2) all asynchronous task has been queued (e.g. NCCL related kernel)
   */
  virtual bool IsKernelLaunchSynchronized() const { return true; }

  void SystemForwardHeader(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    ForwardHeader(ctx, BnInOp2Blob);
  }
  void SystemForwardDataContent(const KernelCtx& ctx,
                                std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    ForwardDataContent(ctx, BnInOp2Blob);
  }
  virtual void Forward(const KernelCtx& ctx,
                       std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  void SetOutputBlobProducerInferAccessChecker(
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void SetOutputBlobProducerComputeAccessChecker(
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void SetOutputBlobConsumerAccessChecker(
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;

 protected:
  Kernel() : job_desc_(nullptr), shape_infer_helper_(nullptr) {}
  void InitBase(const JobDesc* job_desc, const KernelConf&);
  virtual void VirtualKernelInit(DeviceCtx* device_ctx) { VirtualKernelInit(); }
  virtual void VirtualKernelInit() {}
  const KernelConf& kernel_conf() const { return kernel_conf_; }

  virtual void InitConstBufBlobs(DeviceCtx* ctx,
                                 std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

  template<typename HandlerT>
  void ForEachObnAndIsHeaderInferedBeforeCompute(
      std::function<Blob*(const std::string&)> BnInOp2Blob, const HandlerT& Handler) const {
    const auto& modifier_map =
        this->kernel_conf_.op_attribute().arg_modifier_signature().obn2output_blob_modifier();
    for (const std::string& obn : this->op_attribute().output_bns()) {
      Blob* blob = BnInOp2Blob(obn);
      if (blob) {
        bool is_header_infered_before_compute =
            modifier_map.at(obn).header_infered_before_compute();
        Handler(obn, is_header_infered_before_compute);
      }
    }
  }

  template<typename HandlerT>
  void ForEachObnAndIsMutableByConsumer(std::function<Blob*(const std::string&)> BnInOp2Blob,
                                        const HandlerT& Handler) const {
    const auto& modifier_map =
        this->kernel_conf_.op_attribute().arg_modifier_signature().obn2output_blob_modifier();
    for (const std::string& obn : this->op_attribute().output_bns()) {
      Blob* blob = BnInOp2Blob(obn);
      if (blob) {
        bool is_mutable_by_consumer = modifier_map.at(obn).is_mutable();
        Handler(obn, is_mutable_by_consumer);
      }
    }
  }

  virtual void ForwardHeader(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  virtual void ForwardShape(const KernelCtx& ctx,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void NaiveForwardShape(std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  // TODO(niuchong) : rename ForwardDataContent to ForwardBody
  virtual void ForwardDataContent(const KernelCtx& ctx,
                                  std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void ForwardPackedHeader(const KernelCtx& ctx,
                                   std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual bool IsStateless() const { return false; }
  virtual const PbMessage& GetCustomizedOpConf() const { UNIMPLEMENTED(); }
  virtual const PbMessage& GetCustomizedKernelConf() const { UNIMPLEMENTED(); }
  void CheckSameDim0ValidNum(const PbRpf<std::string>& bns,
                             const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;

#define DEFINE_GET_VAL_FROM_CUSTOMIZED_CONF(conf_type)                                   \
  template<typename T>                                                                   \
  T GetValFromCustomized##conf_type(const std::string& field_name) const {               \
    const PbMessage& customized_conf = GetCustomized##conf_type();                       \
    return GetValFromPbMessage<T>(customized_conf, field_name);                          \
  }                                                                                      \
  template<typename T>                                                                   \
  const PbRf<T>& GetPbRfFromCustomized##conf_type(const std::string& field_name) const { \
    return GetPbRfFromPbMessage<T>(GetCustomized##conf_type(), field_name);              \
  }                                                                                      \
  int32_t GetEnumFromCustomized##conf_type(const std::string& field_name) const {        \
    return GetEnumFromPbMessage(GetCustomized##conf_type(), field_name);                 \
  }

  DEFINE_GET_VAL_FROM_CUSTOMIZED_CONF(OpConf);
  DEFINE_GET_VAL_FROM_CUSTOMIZED_CONF(KernelConf);

#undef DEFINE_GET_VAL_FROM_CUSTOMIZED_CONF

 private:
  const JobDesc* job_desc_;
  RuntimeBlobShapeInferHelper* shape_infer_helper_;
  KernelConf kernel_conf_;
};

template<DeviceType device_type>
class KernelIf : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelIf);
  virtual ~KernelIf() = default;

 protected:
  KernelIf() = default;

  virtual void ForwardPackedHeader(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    CopyField(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(), op_attribute().output_bns(),
              &Blob::CopyHeaderFrom);
  }
  void CopyField(DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const Blob* from_blob, const PbRpf<std::string>& to_bns,
                 void (Blob::*Copy)(DeviceCtx*, const Blob*)) const {
    for (const std::string& to_bn : to_bns) { (BnInOp2Blob(to_bn)->*Copy)(ctx, from_blob); }
  }
  void CopyField(DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const PbRpf<std::string>& from_bns, const PbRpf<std::string>& to_bns,
                 void (Blob::*Copy)(DeviceCtx*, const Blob*)) const {
    if (from_bns.size() == 1) {
      const Blob* in_blob = BnInOp2Blob(from_bns[0]);
      CopyField(ctx, BnInOp2Blob, in_blob, to_bns, Copy);
    } else if (to_bns.size() == 1) {
      Blob* in_blob = BnInOp2Blob(from_bns[0]);
      Blob* out_blob = BnInOp2Blob(to_bns[0]);
      (out_blob->*Copy)(ctx, in_blob);
    } else {
      CHECK_EQ(from_bns.size(), to_bns.size());
      FOR_RANGE(size_t, i, 0, from_bns.size()) {
        Blob* in_blob = BnInOp2Blob(from_bns[i]);
        Blob* out_blob = BnInOp2Blob(to_bns[i]);
        (out_blob->*Copy)(ctx, in_blob);
      }
    }
  }

  bool EnableCudnn() const { return op_conf().enable_cudnn(); }
};

#define REGISTER_KERNEL(k, KernelType) \
  REGISTER_CLASS_WITH_ARGS(k, Kernel, KernelType, const KernelConf&)
#define REGISTER_KERNEL_CREATOR(k, f) REGISTER_CLASS_CREATOR(k, Kernel, f, const KernelConf&)

std::unique_ptr<const Kernel> ConstructKernel(const JobDesc* job_desc, const KernelConf&,
                                              DeviceCtx*);

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
    return creators.at(GetHashKey(device_type, kernel_conf.data_type()))();                  \
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
    return creators.at(device_type)();                                                          \
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
    return creators.at(kernel_conf.data_type())();                                      \
  }                                                                                     \
                                                                                        \
  REGISTER_KERNEL_CREATOR(op_type_case, CreateKernel);                                  \
  }

#define ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(op_type_case, kernel_class, data_type_seq)  \
  namespace {                                                                                \
                                                                                             \
  Kernel* OF_PP_CAT(CreateKernel, __LINE__)(const KernelConf& kernel_conf) {                 \
    static const HashMap<std::string, std::function<Kernel*()>> creators = {                 \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY, (kernel_class),          \
                                         DEVICE_TYPE_SEQ, data_type_seq)                     \
            MAKE_KERNEL_CREATOR_ENTRY(kernel_class, DeviceType::kGPU,                        \
                                      (float16, DataType::kFloat16))};                       \
    DeviceType device_type =                                                                 \
        CHECK_JUST(DeviceType4DeviceTag(kernel_conf.op_attribute().op_conf().device_tag())); \
    return creators.at(GetHashKey(device_type, kernel_conf.data_type()))();                  \
  }                                                                                          \
                                                                                             \
  REGISTER_KERNEL_CREATOR(op_type_case, OF_PP_CAT(CreateKernel, __LINE__));                  \
  }

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_H_
