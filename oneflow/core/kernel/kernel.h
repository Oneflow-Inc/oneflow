#ifndef ONEFLOW_CORE_KERNEL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_H_

#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

class Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Kernel);
  virtual ~Kernel() = default;

  void Init(const ParallelContext*, const KernelConf&, DeviceCtx*);

  void InitModelAndConstBuf(const KernelCtx& ctx, const ParallelContext* parallel_ctx,
                            const Snapshot*,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  void Launch(const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  const LogicalBlobId& BnInOp2Lbi(const std::string& bn_in_op) const;
  const OperatorConf& op_conf() const { return op_attribute().op_conf(); }
  const OpAttribute& op_attribute() const { return kernel_conf().op_attribute(); }

 protected:
  Kernel() = default;
  virtual void VirtualKernelInit(const ParallelContext* parallel_ctx, DeviceCtx* device_ctx) {
    VirtualKernelInit(parallel_ctx);
  }
  virtual void VirtualKernelInit(const ParallelContext*) {}
  const KernelConf& kernel_conf() const { return kernel_conf_; }

  virtual void InitConstBufBlobs(DeviceCtx* ctx,
                                 std::function<Blob*(const std::string&)> BnInOp2Blob) const {}
  virtual void InitModelBlobsWithRandomSeed(
      DeviceCtx* ctx, std::mt19937* random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {}
  virtual void InitModelBlobsWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                     const std::string& model_load_dir,
                                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {}
  virtual ActivationType GetActivationType() const { return ActivationType::kNone; }

  virtual void Forward(const KernelCtx& ctx,
                       std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  virtual void ForwardDataContent(const KernelCtx& ctx,
                                  std::function<Blob*(const std::string&)> BnInOp2Blob) const {}
  virtual void ForwardDataId(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual void ForwardColNum(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual void ForwardDim0ValidNum(const KernelCtx& ctx,
                                   std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual void ForwardDim1ValidNum(const KernelCtx& ctx,
                                   std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual void ForwardDim2ValidNum(const KernelCtx& ctx,
                                   std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual void ForwardRecordIdInDevicePiece(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual bool NeedForwardLossInstanceNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual void ForwardLossInstanceNum(const KernelCtx& ctx,
                                      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual void ForwardPackedHeader(const KernelCtx& ctx,
                                   std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual void ForwardActivation(const KernelCtx& ctx, Blob* out_blob) const {}

  virtual void Backward(const KernelCtx& ctx,
                        std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  virtual void BackwardDataContent(const KernelCtx& ctx,
                                   std::function<Blob*(const std::string&)> BnInOp2Blob) const {}
  virtual void BackwardDataId(const KernelCtx& ctx,
                              std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual void BackwardColNum(const KernelCtx& ctx,
                              std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual bool NeedForwardIfBlobEmpty() const { return false; }
  virtual bool NeedBackwardIfBlobEmpty() const { return false; }
  virtual void BackwardInDiffDim0ValidNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual void BackwardInDiffLossInstanceNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual void BackwardModelDiffDim0ValidNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual void BackwardActivation(const KernelCtx& ctx, const Blob* out_blob,
                                  const Blob* out_diff_blob, Blob* bw_activation_blob) const {}
  virtual void SetTotalInstanceNumDiffBlob(
      const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
    UNIMPLEMENTED();
  }
  virtual const PbMessage& GetCustomizedOpConf() const { UNIMPLEMENTED(); }
  virtual const PbMessage& GetCustomizedKernelConf() const { UNIMPLEMENTED(); }
  bool HasEmptyShapeBlob(const PbRpf<std::string>& bns,
                         const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
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
  bool HasModelBns() const;
  KernelConf kernel_conf_;
};

template<DeviceType device_type>
class KernelIf : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelIf);
  virtual ~KernelIf() = default;

 protected:
  KernelIf() = default;

  virtual void ForwardDataId(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void ForwardColNum(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void ForwardDim0ValidNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void ForwardRecordIdInDevicePiece(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void ForwardLossInstanceNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual bool NeedForwardLossInstanceNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void ForwardPackedHeader(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void BackwardDataId(const KernelCtx& ctx,
                              std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void BackwardColNum(const KernelCtx& ctx,
                              std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void BackwardInDiffDim0ValidNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void BackwardInDiffLossInstanceNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void BackwardModelDiffDim0ValidNum(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void CopyField(DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const Blob* from_blob, const PbRpf<std::string>& to_bns,
                 void (Blob::*Copy)(DeviceCtx*, const Blob*)) const;
  void CopyField(DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const PbRpf<std::string>& from_bns, const PbRpf<std::string>& to_bns,
                 void (Blob::*Copy)(DeviceCtx*, const Blob*)) const;

  bool EnableCudnn() const { return op_conf().enable_cudnn(); }
};

template<DeviceType device_type, typename ModelType>
class KernelIfWithModel : virtual public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelIfWithModel);
  virtual ~KernelIfWithModel() = default;

 private:
  void SetTotalInstanceNumDiffBlob(
      const KernelCtx& ctx,
      const std::function<Blob*(const std::string&)>& BnInOp2Blob) const override;

 protected:
  KernelIfWithModel() = default;
};

template<DeviceType device_type, typename T>
class KernelIfWithActivation : virtual public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelIfWithActivation);
  virtual ~KernelIfWithActivation() = default;

 protected:
  KernelIfWithActivation() = default;

  ActivationType GetActivationType() const override;
  void ForwardActivation(const KernelCtx& ctx, Blob* out_blob) const override;
  void BackwardActivation(const KernelCtx& ctx, const Blob* out_blob, const Blob* out_diff_blob,
                          Blob* bw_activation_blob) const override;
};

#define REGISTER_KERNEL(k, KernelType) \
  REGISTER_CLASS_WITH_ARGS(k, Kernel, KernelType, const KernelConf&)
#define REGISTER_KERNEL_CREATOR(k, f) REGISTER_CLASS_CREATOR(k, Kernel, f, const KernelConf&)

std::unique_ptr<const Kernel> ConstructKernel(const ParallelContext*, const KernelConf&,
                                              DeviceCtx*);

}  // namespace oneflow

#define MAKE_KERNEL_CREATOR_ENTRY(kernel_class, device_type, data_type_pair) \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)),               \
   []() { return new kernel_class<device_type, OF_PP_PAIR_FIRST(data_type_pair)>(); }},

#define ADD_DEFAULT_KERNEL_CREATOR(op_type_case, kernel_class, data_type_seq)         \
  namespace {                                                                         \
                                                                                      \
  Kernel* OF_PP_CAT(CreateKernel, __LINE__)(const KernelConf& kernel_conf) {          \
    static const HashMap<std::string, std::function<Kernel*()>> creators = {          \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY, (kernel_class),   \
                                         DEVICE_TYPE_SEQ, data_type_seq)};            \
    return creators.at(GetHashKey(kernel_conf.op_attribute().op_conf().device_type(), \
                                  kernel_conf.data_type()))();                        \
  }                                                                                   \
                                                                                      \
  REGISTER_KERNEL_CREATOR(op_type_case, OF_PP_CAT(CreateKernel, __LINE__));           \
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
    return creators.at(kernel_conf.op_attribute().op_conf().device_type())();                   \
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

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_H_
