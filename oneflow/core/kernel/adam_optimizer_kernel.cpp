#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AdamOptimizerKernel : public KernelIf<device_type> {
 public:
  AdamOptimizerKernel() = default;
  virtual ~AdamOptimizerKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;

  typedef std::function<Blob*(const std::string&)> BnInOp2BlobFunc;
  void Forward(const KernelCtx& ctx,
               BnInOp2BlobFunc BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
void AdamOptimizerKernel<device_type, T>::VirtualKernelInit(
    const ParallelContext*) {
  // TODO(hjchen2)
}

typedef std::function<Blob*(const std::string&)> BnInOp2BlobFunc;
template<DeviceType device_type, typename T>
void AdamOptimizerKernel<device_type, T>::Forward(
    const KernelCtx& ctx, BnInOp2BlobFunc BnInOp2Blob) const {
  // TODO(hjchen2)
  // LOG(INFO) << "Run AdamOptimizerKernel";
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAdamOptimizerConf,
                           AdamOptimizerKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
