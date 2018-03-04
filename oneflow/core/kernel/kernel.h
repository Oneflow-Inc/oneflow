#ifndef ONEFLOW_CORE_KERNEL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_H_

#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

class Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Kernel);
  virtual ~Kernel() = default;

  void Init(const ParallelContext*, const KernelConf&);

  void InitModelAndModelTmp(
      const KernelCtx& ctx, const ParallelContext* parallel_ctx,
      const Snapshot*,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  void Launch(const KernelCtx& ctx,
              std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  const std::string& Lbn4BnInOp(const std::string& bn_in_op) const;

 protected:
  Kernel() = default;
  virtual void VirtualKernelInit(const ParallelContext*) {}
  const KernelConf& kernel_conf() const { return kernel_conf_; }
  const OperatorConf& op_conf() const { return kernel_conf_.op_conf(); }

  virtual void InitModelTmpBlobs(
      DeviceCtx* ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {}
  virtual void InitModelBlobsWithRandomSeed(
      DeviceCtx* ctx, std::mt19937* random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {}
  virtual void InitModelBlobsWithDir(
      DeviceCtx* ctx, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

  virtual void Forward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  virtual void ForwardDataContent(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {}
  virtual void ForwardDataId(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void ForwardColNum(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;

  virtual void Backward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  virtual void BackwardDataContent(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {}
  virtual void BackwardDataId(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void BackwardColNum(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;

  virtual const PbMessage& GetCustomizedOpConf() const { UNIMPLEMENTED(); }
  virtual const PbMessage& GetCustomizedKernelConf() const { UNIMPLEMENTED(); }

#define DEFINE_GET_VAL_FROM_SPECIAL_CONF(ret_type, func_name, conf_type) \
  ret_type Get##func_name##FromCustomized##conf_type(                    \
      const std::string& field_name) const {                             \
    const PbMessage& special_conf = GetCustomized##conf_type();          \
    return Get##func_name##FromPbMessage(special_conf, field_name);      \
  }

#define DEFINE_GET_VAL_FROM_SPECIAL_CONF_TYPE(ret_type, func_name) \
  DEFINE_GET_VAL_FROM_SPECIAL_CONF(ret_type, func_name, OpConf);   \
  DEFINE_GET_VAL_FROM_SPECIAL_CONF(ret_type, func_name, KernelConf);

  OF_PP_FOR_EACH_TUPLE(DEFINE_GET_VAL_FROM_SPECIAL_CONF_TYPE,
                       PROTOBUF_BASIC_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(
                           const PbMessage&, Message));

  template<typename T>
  const PbRf<T>& GetPbRfFromCustomizedOpConf(
      const std::string& field_name) const {
    return GetPbRfFromPbMessage<T>(GetCustomizedOpConf(), field_name);
  }

  template<typename T>
  const PbRf<T>& GetRbRfFromCustomizedKernelConf(
      const std::string& field_name) const {
    return GetPbRfFromPbMessage<T>(GetCustomizedKernelConf(), field_name);
  }

 private:
  KernelConf kernel_conf_;
};

template<DeviceType device_type>
class KernelIf : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelIf);
  virtual ~KernelIf() = default;

 protected:
  KernelIf() = default;

  virtual void ForwardDataId(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void ForwardColNum(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void BackwardDataId(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  virtual void BackwardColNum(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void CopyDataId(DeviceCtx* ctx,
                  std::function<Blob*(const std::string&)> BnInOp2Blob,
                  const Blob* from_blob,
                  const PbRpf<std::string>& to_bns) const;
  void CopyColNum(DeviceCtx* ctx,
                  std::function<Blob*(const std::string&)> BnInOp2Blob,
                  const Blob* from_blob,
                  const PbRpf<std::string>& to_bns) const;
  void CopyField(DeviceCtx* ctx,
                 std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const Blob* from_blob, const PbRpf<std::string>& to_bns,
                 void (Blob::*Copy)(DeviceCtx*, const Blob*)) const;
  void CopyField(DeviceCtx* ctx,
                 std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const PbRpf<std::string>& from_bns,
                 const PbRpf<std::string>& to_bns,
                 void (Blob::*Copy)(DeviceCtx*, const Blob*)) const;
};

using KernelCreator1 = std::function<Kernel*(const KernelConf&)>;
using KernelCreator2 = std::function<Kernel*()>;
void AddKernelCreator(OperatorConf::OpTypeCase, KernelCreator1);
void AddKernelCreator(OperatorConf::OpTypeCase, KernelCreator2);
std::unique_ptr<const Kernel> ConstructKernel(const ParallelContext*,
                                              const KernelConf&);

}  // namespace oneflow

#define MAKE_KERNEL_CREATOR_ENTRY(kernel_class, device_type, data_type_pair)   \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)), []() {          \
     return new kernel_class<device_type, OF_PP_PAIR_FIRST(data_type_pair)>(); \
   }},

#define ADD_DEFAULT_KERNEL_CREATOR(op_type_case, kernel_class, data_type_seq) \
  namespace {                                                                 \
                                                                              \
  Kernel* CreateKernel(const KernelConf& kernel_conf) {                       \
    static const HashMap<std::string, std::function<Kernel*()>> creators = {  \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY,           \
                                         (kernel_class), DEVICE_TYPE_SEQ,     \
                                         data_type_seq)};                     \
    return creators.at(                                                       \
        GetHashKey(kernel_conf.device_type(), kernel_conf.data_type()))();    \
  }                                                                           \
                                                                              \
  COMMAND(AddKernelCreator(op_type_case, CreateKernel));                      \
  }

#define MAKE_CUDNN_KERNEL_CREATOR_ENTRY(kernel_class, data_type_pair) \
  {OF_PP_PAIR_SECOND(data_type_pair),                                 \
   []() { return new kernel_class<OF_PP_PAIR_FIRST(data_type_pair)>(); }},

#define ADD_DEFAULT_CUDNN_KERNEL_CREATOR(op_type_case, kernel_type_field, \
                                         kernel_class, data_type_seq)     \
  template<>                                                              \
  bool CudnnKernelCreatorHelper<op_type_case>::IsUseCudnn(                \
      const KernelConf& kernel_conf) {                                    \
    CHECK(kernel_conf.has_##kernel_type_field());                         \
    if (kernel_conf.kernel_type_field().has_use_cudnn()) {                \
      return kernel_conf.kernel_type_field().use_cudnn();                 \
    } else {                                                              \
      return fasle;                                                       \
    }                                                                     \
  }                                                                       \
  template<>                                                              \
  Kernel* CudnnKernelCreatorHelper<op_type_case>::CreateKernel(           \
      DeviceType dev_type, const KernelConf& kernel_conf) {               \
    static const HashMap<int, std::function<Kernel*()>> creators = {      \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_CUDNN_KERNEL_CREATOR_ENTRY, \
                                         (kernel_class), data_type_seq)}; \
    return creators.at(kernel_conf.data_type())();                        \
  }

#define MAKE_CPU_KERNEL_CREATOR_ENTRY(kernel_class, data_type_pair) \
  {OF_PP_PAIR_SECOND(data_type_pair),                               \
   []() { return new kernel_class<OF_PP_PAIR_FIRST(data_type_pair)>(); }},

#define ADD_CPU_DEFAULT_KERNEL_CREATOR(op_type_case, kernel_class,        \
                                       data_type_seq)                     \
  namespace {                                                             \
                                                                          \
  Kernel* CreateKernel(const KernelConf& kernel_conf) {                   \
    static const HashMap<int, std::function<Kernel*()>> creators = {      \
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_CPU_KERNEL_CREATOR_ENTRY,   \
                                         (kernel_class), data_type_seq)}; \
    return creators.at(kernel_conf.data_type())();                        \
  }                                                                       \
                                                                          \
  COMMAND(AddKernelCreator(op_type_case, CreateKernel));                  \
  }

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_H_
