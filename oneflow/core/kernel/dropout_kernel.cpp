#include "oneflow/core/kernel/dropout_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void DropoutKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t elem_cnt = BnInOp2Blob("in")->shape().elem_cnt();
  if (JobDesc::Singleton()->IsTrain()) {
    KernelUtil<device_type, T>::Dropout(
        ctx.device_ctx, elem_cnt, this->op_conf().dropout_conf().keep_prob(),
        BnInOp2Blob("in")->dptr<T>(),
        BnInOp2Blob("random_mask")->mut_dptr<float>(),
        BnInOp2Blob("out")->mut_dptr<T>());
  } else {
    Memcpy<device_type>(ctx.device_ctx, BnInOp2Blob("out")->mut_dptr<void>(),
                        BnInOp2Blob("in")->dptr<void>(), elem_cnt * sizeof(T));
  }
}

template<DeviceType device_type, typename T>
void DropoutKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::DropoutBackward(
      ctx.device_ctx, BnInOp2Blob("in")->shape().elem_cnt(),
      this->op_conf().dropout_conf().keep_prob(),
      BnInOp2Blob("out_diff")->dptr<T>(),
      BnInOp2Blob("random_mask")->dptr<float>(),
      BnInOp2Blob("in_diff")->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kDropoutConf, DropoutKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
