#include "oneflow/core/kernel/kernel.h"
//#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type>
class ZerosLikeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ZerosLikeKernel);
  ZerosLikeKernel() = default;
  ~ZerosLikeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* out_blob = BnInOp2Blob("y");
    Memset<device_type>(ctx.device_ctx, out_blob->mut_dptr(), 0,
                        out_blob->ByteSizeOfDataContentField());
  }
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().zeros_like_conf();
  }
};

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kZerosLikeConf, DeviceType::kCPU,
                            ZerosLikeKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kZerosLikeConf, DeviceType::kGPU,
                            ZerosLikeKernel<DeviceType::kGPU>);

}  // namespace oneflow
