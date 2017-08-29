#include "oneflow/core/kernel/clone_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void CloneKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(op()->SoleIbn());
  for (const std::string& obn : op()->output_bns()) {
    Blob* out_blob = BnInOp2Blob(obn);
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr(), in_blob->dptr(),
                        in_blob->TotalByteSize());
  }
}

template<DeviceType device_type, typename T>
class CloneKernelUtil final {
 public:
  // b += a
  static void AdditionAssign(DeviceCtx* device_ctx, const Blob* a, Blob* b);
};

template<DeviceType device_type, typename T>
void CloneKernel<device_type, T>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const std::vector<std::string>& odbns = op()->output_diff_bns();
  if (odbns.size() == 0) return;
  Blob* idbn_blob = BnInOp2Blob(op()->SoleIdbn());
  const Blob* odbn_blob_0 = BnInOp2Blob(odbns[0]);
  Memcpy<device_type>(ctx.device_ctx, idbn_blob->mut_dptr(),
                      odbn_blob_0->dptr(), odbn_blob_0->TotalByteSize());
  for (size_t i = 1; i != odbns.size(); ++i) {
    const Blob* odbn_blob = BnInOp2Blob(odbns[i]);
    CloneKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, odbn_blob,
                                                    idbn_blob);
  }
}

#define DEFINE_FLOATING_CLONE_KERNEL_UTIL(type_cpp, type_proto)           \
  template<DeviceType device_type>                                        \
  class CloneKernelUtil<device_type, type_cpp> final {                    \
   public:                                                                \
    static void AdditionAssign(DeviceCtx* device_ctx, const Blob* a,      \
                               Blob* b) {                                 \
      KernelUtil<device_type, type_cpp>::BlasAxpy(                        \
          device_ctx, a->shape().elem_cnt(), 1.0, a->dptr<type_cpp>(), 1, \
          b->mut_dptr<type_cpp>(), 1);                                    \
    }                                                                     \
  };

FOR_EACH_PAIR(DEFINE_FLOATING_CLONE_KERNEL_UTIL, FLOATING_DATA_TYPE_PAIR())

#define DEFINE_NONFLOAT_CLONE_KERNEL_UTIL(type_cpp, type_proto)      \
  template<DeviceType device_type>                                   \
  class CloneKernelUtil<device_type, type_cpp> final {               \
   public:                                                           \
    static void AdditionAssign(DeviceCtx* device_ctx, const Blob* a, \
                               Blob* b) {                            \
      UNEXPECTED_RUN();                                              \
    }                                                                \
  };

FOR_EACH_PAIR(DEFINE_NONFLOAT_CLONE_KERNEL_UTIL,
              INT_DATA_TYPE_PAIR() CHAR_DATA_TYPE_PAIR())

namespace {

template<DeviceType device_type>
Kernel* CreateCloneKernel(const OperatorConf& op_conf) {
  static const HashMap<int, std::function<Kernel*()>> data_type2creator = {
#define CLONE_KERNEL_ENTRY(type_cpp, type_proto) \
  {type_proto, []() { return new CloneKernel<device_type, type_cpp>; }},
      FOR_EACH_PAIR(CLONE_KERNEL_ENTRY, ALL_DATA_TYPE_PAIR())};
  return data_type2creator.at(op_conf.clone_conf().data_type())();
}

}  // namespace

REGISTER_TEMPLATE_KERNEL_CREATOR(OperatorConf::kCloneConf, CreateCloneKernel);

}  // namespace oneflow
