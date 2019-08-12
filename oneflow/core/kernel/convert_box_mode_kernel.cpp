#include "oneflow/core/kernel/convert_box_mode_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConvertBoxModeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<typename T>
struct ConvertBoxModeUtil<DeviceType::kCPU, T> {
  static void Forward() {
    // TODO
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConvertBoxModeConf, ConvertBoxModeKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
