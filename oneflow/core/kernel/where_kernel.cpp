#include "oneflow/core/kernel/where_kernel.h"
#include "oneflow/core/kernel/where_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void WhereKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* condition_blob = BnInOp2Blob("condition");
  int64_t elem_cnt = condition_blob->shape().elem_cnt();

  WhereKernelUtil<device_type, T>::Where(ctx.device_ctx, elem_cnt, condition_blob->dptr<T>(),
                                         BnInOp2Blob("x")->dptr<T>(), BnInOp2Blob("y")->dptr<T>(),
                                         BnInOp2Blob("out")->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kWhereConf, WhereKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
