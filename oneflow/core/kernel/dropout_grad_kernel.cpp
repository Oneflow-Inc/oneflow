#include "oneflow/core/kernel/dropout_kernel.h"
#include "oneflow/core/kernel/dropout_grad_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/preprocessor.h"
namespace oneflow {

template<DeviceType device_type, typename T>
void DropoutGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  DropoutKernelUtil<device_type, T>::MaskAndScale(
      ctx.device_ctx, BnInOp2Blob("dy")->shape().elem_cnt(),
      this->op_conf().dropout_grad_conf().scale(), BnInOp2Blob("dy")->dptr<T>(),
      BnInOp2Blob("mask")->dptr<int8_t>(), BnInOp2Blob("dx")->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kDropoutGradConf, DropoutGradKernel,
                           ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);

}  // namespace oneflow
