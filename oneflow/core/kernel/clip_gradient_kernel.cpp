#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ClipGradientKernel : public KernelIf<device_type> {
 public:
  ClipGradientKernel() = default;
  virtual ~ClipGradientKernel() = default;

 private:
  typedef std::function<Blob*(const std::string&)> BnInOp2BlobFunc;
  void Forward(const KernelCtx& ctx,
               BnInOp2BlobFunc BnInOp2Blob) const override;
};

typedef std::function<Blob*(const std::string&)> BnInOp2BlobFunc;
template<DeviceType device_type, typename T>
void ClipGradientKernel<device_type, T>::Forward(
    const KernelCtx& ctx, BnInOp2BlobFunc BnInOp2Blob) const {
  // LOG(INFO) << "Run ClipGradientKernel";
  // TODO(hjchen2)
  LOG(FATAL) << "ClipGradient is only used for XLA.";
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kClipGradientConf, ClipGradientKernel,
                           FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
