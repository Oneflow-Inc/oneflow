#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<typename FloatingPointType>
void PoolingKernel<DeviceType::kCPU, FloatingPointType>::Forward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  Blob* in_blob = bn_in_op2blob_ptr("in");
  Blob* out_blob = bn_in_op2blob_ptr("out");
  PoolingOpConf_PoolMethod pool_method = op()->op_conf().pooling_conf().pool();
  switch (pool_method) {
    case PoolingOpConf_PoolMethod_MAX: break;
    case PoolingOpConf_PoolMethod_AVE: break;
    case PoolingOpConf_PoolMethod_STOCHASTIC: break;
  }
}

template<typename FloatingPointType>
void PoolingKernel<DeviceType::kCPU, FloatingPointType>::Backward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  TODO();
}

INSTANTIATE_CPU_KERNEL_CLASS(PoolingKernel);
REGISTER_CPU_KERNEL(OperatorConf::kPoolingConf, PoolingKernel);

}  // namespace oneflow
