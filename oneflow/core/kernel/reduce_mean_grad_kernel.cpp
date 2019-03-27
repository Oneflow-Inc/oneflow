#include "oneflow/core/kernel/reduce_mean_grad_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceMeanGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* dy_blob = BnInOp2Blob("dy");
  Blob* dx_blob = BnInOp2Blob("dx");
  Blob* tmp_blob = BnInOp2Blob("temp_storage");
  const Blob* const_tmp_blob = const_tmp_blob;
  size_t count = dx_blob->shape().elem_cnt() / dy_blob->shape().elem_cnt();
  Memcpy<device_type>(ctx.device_ctx, tmp_blob->mut_dptr(), dy_blob->dptr(),
                      dy_blob->ByteSizeOfDataContentField());
  KernelUtil<device_type, T>::Div(ctx.device_ctx, tmp_blob->shape().elem_cnt(),
                                  tmp_blob->mut_dptr<T>(), static_cast<T>(count));
  const int64_t num_axes = dx_blob->shape().NumAxes();
  if (this->op_conf().reduce_mean_grad_conf().has_kept_dims_shape()) {
    NdarrayUtil<device_type, T>::BroadcastTo(
        ctx.device_ctx, XpuVarNdarray<T>(dx_blob, num_axes),
        XpuVarNdarray<const T>(Shape(this->op_conf().reduce_mean_grad_conf().kept_dims_shape()),
                               tmp_blob->dptr<T>()));
  } else {
    NdarrayUtil<device_type, T>::BroadcastTo(ctx.device_ctx, XpuVarNdarray<T>(dx_blob, num_axes),
                                             XpuVarNdarray<const T>(const_tmp_blob, num_axes));
  }
}

template<DeviceType device_type, typename T>
void ReduceMeanGradKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("dx")->CopyDim0ValidNumFrom(ctx.device_ctx, BnInOp2Blob("x"));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceMeanGradConf, ReduceMeanGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
