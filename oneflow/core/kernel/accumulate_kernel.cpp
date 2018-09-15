#include "oneflow/core/kernel/accumulate_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void AccumulateKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("one");
  Blob* out_blob = BnInOp2Blob("acc");
  KernelUtil<device_type, T>::Axpy(ctx.device_ctx, in_blob->shape().elem_cnt(), OneVal<T>::value,
                                   in_blob->dptr<T>(), 1, out_blob->mut_dptr<T>(), 1);
}

template<DeviceType device_type, typename T>
void AccumulateKernel<device_type, T>::ForwardInstanceNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  this->AccumulateField(ctx.device_ctx, BnInOp2Blob, this->op_attribute().input_bns(),
                        this->op_attribute().output_bns(), &Blob::AccumulateInstanceNumFrom);
  // xfjiang: test instance num
  Blob* one_blob = BnInOp2Blob("one");
  Blob* acc_blob = BnInOp2Blob("acc");
  LOG(INFO) << "acc instance num (one): " << *(one_blob->instance_num());
  LOG(INFO) << "acc instance num (acc): " << *(acc_blob->instance_num());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAccumulateConf, AccumulateKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
