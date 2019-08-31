#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LearningRateShedulerKernel : public KernelIf<device_type> {
 public:
  LearningRateShedulerKernel() = default;
  virtual ~LearningRateShedulerKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;

  typedef std::function<Blob*(const std::string&)> BnInOp2BlobFunc;
  void Forward(const KernelCtx& ctx,
               BnInOp2BlobFunc BnInOp2Blob) const override;

  float base_learning_rate_;
};

template<DeviceType device_type, typename T>
void LearningRateShedulerKernel<device_type, T>::VirtualKernelInit(
    const ParallelContext*) {
  // TODO(hjchen2)
}

typedef std::function<Blob*(const std::string&)> BnInOp2BlobFunc;

template<DeviceType device_type, typename T>
void LearningRateShedulerKernel<device_type, T>::Forward(
    const KernelCtx& ctx, BnInOp2BlobFunc BnInOp2Blob) const {
  // LOG(INFO) << "Run LearningRateShedulerKernel";
  // TODO(hjchen2)
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLrShedulerConf,
                           LearningRateShedulerKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
