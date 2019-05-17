#include "oneflow/core/kernel/pooling_grad_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PoolingGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* dx_blob = BnInOp2Blob("dx");
  if (dx_blob == nullptr) { return; }
  const PoolingConf& pooling_conf = this->op_conf().pooling_grad_conf().pooling_conf();
  if (pooling_conf.pool_mode() == "max") {
    Memset<device_type>(ctx.device_ctx, dx_blob->mut_dptr(), 0,
                        dx_blob->ByteSizeOfDataContentField());
  }
  PoolingGradKernelUtil<device_type, T>::Compute(ctx.device_ctx, pooling_conf, BnInOp2Blob("dy"),
                                                 BnInOp2Blob("y"), BnInOp2Blob("x"), dx_blob);
}

template<DeviceType device_type, typename T>
const PbMessage& PoolingGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().pooling_grad_conf();
}

template<typename T>
struct PoolingGradKernelUtil<DeviceType::kCPU, T> final {
  static void Compute(DeviceCtx* ctx, const PoolingConf& pooling_conf, const Blob* dy_blob,
                      const Blob* y_blob, const Blob* x_blob, Blob* dx_blob) {
    UNIMPLEMENTED();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kPoolingGradConf, PoolingGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
