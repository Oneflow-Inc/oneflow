#include "oneflow/core/kernel/total_loss_instance_num_kernel.h"

namespace oneflow {

template<typename T>
void TotalLossInstanceNumKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& input_bns = this->op_attribute().input_bns();
  for (const std::string& ibn : input_bns) {
    CHECK_EQ(*BnInOp2Blob(ibn)->template dptr<T>(),
             *BnInOp2Blob(input_bns.Get(0))->template dptr<T>());
  }
  *BnInOp2Blob("out")->template mut_dptr<T>() = *BnInOp2Blob(input_bns.Get(0))->template dptr<T>();
}

#define REGISTER_TOTALLOSSINSTANCENUM_KERNEL(dtype)                                                \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kTotalLossInstanceNumConf, DeviceType::kCPU, \
                                        dtype, TotalLossInstanceNumKernel<dtype>);

REGISTER_TOTALLOSSINSTANCENUM_KERNEL(int32_t);
REGISTER_TOTALLOSSINSTANCENUM_KERNEL(int64_t);
REGISTER_TOTALLOSSINSTANCENUM_KERNEL(float);
REGISTER_TOTALLOSSINSTANCENUM_KERNEL(double);

}  // namespace oneflow
