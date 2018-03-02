#include "oneflow/core/kernel/eltwise_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace oneflow {

template<DeviceType device_type, typename T>
void EltwiseKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const EltwiseOpConf& eltwise_conf = this->op_conf().eltwise_conf();
  Blob* out_blob = BnInOp2Blob("out");
  Memset<device_type>(ctx.device_ctx, out_blob->mut_dptr(), 0,
                      out_blob->ByteSizeOfDataContentField());
  switch (eltwise_conf.operation()) {
    case EltwiseOpConf_EltwiseOp_SUM:
      // out = sum(in1, in2, in3....)
      for (int i = 0; i < eltwise_conf.in_size(); ++i) {
        std::string ibn = "in_" + std::to_string(i);
        const Blob* in_blob = BnInOp2Blob(ibn);
        KernelUtil<device_type, T>::Axpy(
            ctx.device_ctx, in_blob->shape().elem_cnt(), 1.0f,
            in_blob->dptr<T>(), 1, out_blob->mut_dptr<T>(), 1);
      }
    case EltwiseOpConf_EltwiseOp_MAX:
      // out = max(in1, in2, in3....)
      for (int i = 0; i < eltwise_conf.in_size(); ++i) {
        std::string ibn = "in_" + std::to_string(i);
        const Blob* in_blob = BnInOp2Blob(ibn);
      };
    default: break;
  }
}

template<DeviceType device_type, typename T>
void EltwiseKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kEltwiseConf, EltwiseKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
