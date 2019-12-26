#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class LogisticKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogisticKernel);
  LogisticKernel() = default;
  ~LogisticKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    KernelUtil<device_type, T>::Sigmoid(ctx.device_ctx, in_blob->shape().elem_cnt(),
                                        in_blob->dptr<T>(), BnInOp2Blob("out")->mut_dptr<T>());
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLogisticConf, LogisticKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
