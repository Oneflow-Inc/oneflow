#include "oneflow/core/kernel/clone_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void CloneKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(this->kernel_conf().input_bns(0));
  for (const std::string& obn : this->kernel_conf().output_bns()) {
    Blob* out_blob = BnInOp2Blob(obn);
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_memory_ptr(),
                        in_blob->memory_ptr(), in_blob->TotalByteSize());
  }
}

template<DeviceType device_type, typename T>
class CloneKernelUtil final {
 public:
  // b += a
  static void AdditionAssign(DeviceCtx* device_ctx, const Blob* a, Blob* b);
};

template<DeviceType device_type, typename T>
void CloneKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& odbns = this->kernel_conf().output_diff_bns();
  if (odbns.size() == 0) return;
  Blob* in_diff_blob = BnInOp2Blob(this->kernel_conf().input_diff_bns(0));
  const Blob* out_diff_blob_0 = BnInOp2Blob(odbns[0]);
  Memcpy<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(),
                      out_diff_blob_0->dptr(),
                      out_diff_blob_0->ByteSizeOfDataContentField());
  for (size_t i = 1; i != odbns.size(); ++i) {
    const Blob* out_diff_blob = BnInOp2Blob(odbns[i]);
    CloneKernelUtil<device_type, T>::AdditionAssign(
        ctx.device_ctx, out_diff_blob, in_diff_blob);
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

OF_PP_FOR_EACH_TUPLE(DEFINE_FLOATING_CLONE_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

#define DEFINE_NONFLOAT_CLONE_KERNEL_UTIL(type_cpp, type_proto)      \
  template<DeviceType device_type>                                   \
  class CloneKernelUtil<device_type, type_cpp> final {               \
   public:                                                           \
    static void AdditionAssign(DeviceCtx* device_ctx, const Blob* a, \
                               Blob* b) {                            \
      UNEXPECTED_RUN();                                              \
    }                                                                \
  };

OF_PP_FOR_EACH_TUPLE(DEFINE_NONFLOAT_CLONE_KERNEL_UTIL,
                     INT_DATA_TYPE_SEQ CHAR_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kCloneConf, CloneKernel,
                           ALL_DATA_TYPE_SEQ);

}  // namespace oneflow
