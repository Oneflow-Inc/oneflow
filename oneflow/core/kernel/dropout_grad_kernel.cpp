#include "oneflow/core/kernel/dropout_kernel.h"
#include "oneflow/core/kernel/dropout_grad_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/preprocessor.h"
namespace oneflow {

template<DeviceType device_type, typename T>
void DropoutGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  DropoutBackward(ctx.device_ctx, BnInOp2Blob("dy")->shape().elem_cnt(),
                  this->op_conf().dropout_conf().rate(), BnInOp2Blob("dy")->dptr<T>(),
                  BnInOp2Blob("random_mask")->dptr<float>(), BnInOp2Blob("dx")->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void DropoutGradKernel<device_type, T>::DropoutBackward(DeviceCtx* ctx, const int64_t n,
                                                        float dropout_rate, const T* dy,
                                                        const float* random_mask, T* dx) const {
  DropoutKernelUtil<device_type, T>::MaskAndScale(ctx, n, dropout_rate, 1 / (1 - dropout_rate), dy,
                                                  random_mask, dx);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kDropoutGradConf, DropoutGradKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
