#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/nvtx3/nvToolsExt.h"

namespace oneflow {

template<DeviceType device_type>
class NvtxRangePushKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NvtxRangePushKernel);
  NvtxRangePushKernel() = default;
  ~NvtxRangePushKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx &ctx,
                          std::function<Blob *(const std::string &)> BnInOp2Blob) const override {
    nvtxRangePush(this->op_conf().nvtx_range_push_conf().msg().c_str());
  }
};

template<DeviceType device_type>
class NvtxRangePopKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NvtxRangePopKernel);
  NvtxRangePopKernel() = default;
  ~NvtxRangePopKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx &ctx,
                          std::function<Blob *(const std::string &)> BnInOp2Blob) const override {
    nvtxRangePop();
  }
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kNvtxRangePushConf, NvtxRangePushKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kNvtxRangePopConf, NvtxRangePopKernel);

}  // namespace oneflow
