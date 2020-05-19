#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/square_sum_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SquareSumKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SquareSumKernel);
  SquareSumKernel() = default;
  ~SquareSumKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
void SquareSumKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* x = BnInOp2Blob("x");
  Blob* y = BnInOp2Blob("y");
  SquareSumKernelUtil<device_type, T>::SquareSum(ctx.device_ctx, x->shape().elem_cnt(),
                                                 x->dptr<T>(), y->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSquareSumConf, SquareSumKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
