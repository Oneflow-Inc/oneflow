#include "oneflow/core/kernel/reduce_sum_like_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceSumLikeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* x_blob = BnInOp2Blob("x");
  Blob* y_blob = BnInOp2Blob("y");
  Blob* temp_storage_blob = BnInOp2Blob("temp_storage");
  const ReduceSumLikeOpConf& conf = this->op_conf().reduce_sum_like_conf();
  if (conf.axis().empty()) {
    CHECK_EQ(x_blob->shape(), y_blob->shape());
    y_blob->CopyDataContentFrom(ctx.device_ctx, x_blob);
  } else {
    NdarrayUtil<device_type, T>::ReduceSum(
        ctx.device_ctx,
        XpuVarNdarray<T>(
            CreateReducedShape(x_blob->shape(), {conf.axis().begin(), conf.axis().end()}),
            y_blob->mut_dptr<T>()),
        XpuVarNdarray<const T>(x_blob, x_blob->shape().NumAxes()),
        XpuVarNdarray<T>(temp_storage_blob, x_blob->shape().NumAxes()));
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceSumLikeConf, ReduceSumLikeKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
