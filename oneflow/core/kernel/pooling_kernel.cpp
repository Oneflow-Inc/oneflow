#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

namespace {

template<FloatingPointType>
std::pair<size_t, FloatingPointType> GetMaxValue(
    FloatingPointType* dptr, size_t pad_height, size_t pad_weight) {
  
}

template<FloatingPointType>
std::pair<size_t, FloatingPointType> GetAveValue() {
  
}

template<FloatingPointType>
std::pair<size_t, FloatingPointType> GetStoValue();

}  // namespace

template<typename FloatingPointType>
void PoolingKernel<DeviceType::kCPU, FloatingPointType>::InitFromOpProto(
    const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);

  switch(op()->op_conf().pooling_conf().pool()) {
    case PoolingOpConf::MAX:
      PoolingMethodFunc_ = &GetMaxValue;
      break;
    case PoolingOpConf::AVE:
      PoolingMethodFunc_ = &GetAveValue;
      break;
    case PoolingOpConf::STOCHASTIC:
      PoolingMethodFunc_ = &GetStoValue;
      break;
    default:
      UNEXPECTED_RUN();
  }
}

template<typename FloatingPointType>
void PoolingKernel<DeviceType::kCPU, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* data_tmp = BnInOp2Blob("idx");
    
}

template<typename FloatingPointType>
void PoolingKernel<DeviceType::kCPU, FloatingPointType>::Backward(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> bn_in_op2blob_ptr) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  Blob* data_temp = BnInOp2Blob("idx");

}

INSTANTIATE_CPU_KERNEL_CLASS(PoolingKernel);
REGISTER_CPU_KERNEL(OperatorConf::kPoolingConf, PoolingKernel);

}  // namespace oneflow
