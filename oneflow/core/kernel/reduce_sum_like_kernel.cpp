#include "oneflow/core/kernel/reduce_sum_like_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceSumLikeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* x_blob = BnInOp2Blob("x");
  Blob* y_blob = BnInOp2Blob("y");
  Blob* fw_tmp_blob = BnInOp2Blob("fw_tmp");
  NdarrayUtil<device_type, T>::ReduceSum(
      ctx.device_ctx,
      XpuVarNdarray<T>(
          Shape(BnInOp2Blob("like")->shape()).CreateLeftExtendedShape(x_blob->shape().NumAxes()),
          y_blob->mut_dptr<T>()),
      XpuVarNdarray<const T>(x_blob, x_blob->shape().NumAxes()),
      XpuVarNdarray<T>(fw_tmp_blob, x_blob->shape().NumAxes()));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceSumLikeConf, ReduceSumLikeKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
