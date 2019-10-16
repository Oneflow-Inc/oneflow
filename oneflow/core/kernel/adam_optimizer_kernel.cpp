#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AdamOptimizerKernel : public KernelIf<device_type> {
 public:
  AdamOptimizerKernel() = default;
  virtual ~AdamOptimizerKernel() = default;

 private:
  void VirtualKernelInit() override;

  typedef std::function<Blob*(const std::string&)> BnInOp2BlobFunc;
  void Forward(const KernelCtx& ctx, BnInOp2BlobFunc BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
void AdamOptimizerKernel<device_type, T>::VirtualKernelInit() {
  // TODO(hjchen2)
  LOG(FATAL) << "AdamOptimizer is only used for XLA.";
}

typedef std::function<Blob*(const std::string&)> BnInOp2BlobFunc;
template<DeviceType device_type, typename T>
void AdamOptimizerKernel<device_type, T>::Forward(const KernelCtx& ctx,
                                                  BnInOp2BlobFunc BnInOp2Blob) const {
  // TODO(hjchen2)
  LOG(FATAL) << "AdamOptimizer is only used for XLA.";
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAdamOptimizerConf, AdamOptimizerKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
