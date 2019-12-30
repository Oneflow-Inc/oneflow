#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type>
class Identity1Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Identity1Kernel);
  Identity1Kernel() = default;
  ~Identity1Kernel() = default;

 private:
  void ForwardDataContent(const KernelCtx &,
                          std::function<Blob *(const std::string &)>) const override;
};

template<DeviceType device_type>
void Identity1Kernel<device_type>::ForwardDataContent(
    const KernelCtx &ctx, std::function<Blob *(const std::string &)> BnInOp2Blob) const {}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kIdentity1Conf, Identity1Kernel);

}  // namespace oneflow
