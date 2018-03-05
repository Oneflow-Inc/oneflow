#include "oneflow/core/kernel/eltwise_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace oneflow {

template<DeviceType device_type, typename T>
void EltwiseKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const EltwiseOpConf& eltwise_conf = this->op_conf().eltwise_conf();
  Blob* out_blob = BnInOp2Blob("out");
  const Blob* in_blob0 = BnInOp2Blob("in_0");
  Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(),
                      in_blob0->dptr<T>(),
                      out_blob->ByteSizeOfDataContentField());
  Blob* mask_blob = BnInOp2Blob("mask");
  // set mask to zero because out is set to in0
  Memset<device_type>(ctx.device_ctx, mask_blob->mut_dptr(), 0,
                      mask_blob->ByteSizeOfDataContentField());
  const int count = out_blob->shape().elem_cnt();
  switch (eltwise_conf.operation()) {
    case EltwiseOpConf_EltwiseOp_SUM:
      // out = out + sum(in1, in2, in3....)
      // start at index of 1 because out is set to in0
      for (int i = 1; i < eltwise_conf.in_size(); ++i) {
        std::string ibn = "in_" + std::to_string(i);
        const Blob* in_blob = BnInOp2Blob(ibn);
        KernelUtil<device_type, T>::Axpy(ctx.device_ctx, count, 1.0f,
                                         in_blob->dptr<T>(), 1,
                                         out_blob->mut_dptr<T>(), 1);
      }
      break;
    case EltwiseOpConf_EltwiseOp_MAX:
      // out = max(in0, in1, in2, in3....)
      // mask = index_of_max(in0, in1, in2, in3....)
      // start at index of 1 because out is set to in0
      for (int i = 1; i < eltwise_conf.in_size(); ++i) {
        std::string ibn = "in_" + std::to_string(i);
        const Blob* in_blob = BnInOp2Blob(ibn);
        for (int idx = 0; idx < count; ++idx) {
          if (in_blob->dptr<T>()[idx] > out_blob->mut_dptr<T>()[idx]) {
            out_blob->mut_dptr<T>()[idx] = in_blob->dptr<T>()[idx];
            mask_blob->mut_dptr<T>()[idx] = i;
          }
        }
      }
      break;
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
