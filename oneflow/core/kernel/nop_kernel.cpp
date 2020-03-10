#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class NopKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NopKernel);
  NopKernel() = default;
  ~NopKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kAccTickConf, NopKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kCopyCommNetConf, NopKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kDeviceTickConf, NopKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kInputConf, NopKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kKeepHeaderOnlyConf, NopKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kPartialTickConf, NopKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kSinkTickConf, NopKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kSourceTickConf, NopKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kTickConf, NopKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kVariableConf, NopKernel);

}  // namespace oneflow
