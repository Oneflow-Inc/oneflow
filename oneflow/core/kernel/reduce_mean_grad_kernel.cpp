#include "oneflow/core/kernel/reduce_mean_grad_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceMeanGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* dy_blob = BnInOp2Blob("dy");
  Blob* dx_blob = BnInOp2Blob("dx");
  Blob* tmp_blob = BnInOp2Blob("temp_storage");
  size_t count = dx_blob->shape().elem_cnt() / dy_blob->shape().elem_cnt();
  Memcpy<device_type>(ctx.device_ctx, tmp_blob->mut_dptr(), dy_blob->dptr(),
                      dy_blob->ByteSizeOfDataContentField());
  KernelUtil<device_type, T>::Div(ctx.device_ctx, tmp_blob->shape().elem_cnt(),
                                  tmp_blob->mut_dptr<T>(), static_cast<T>(count));
  NdarrayUtil<device_type, T>::BroadcastTo(
      ctx.device_ctx, XpuVarNdarray<T>(dx_blob, dx_blob->shape().NumAxes()),
      XpuVarNdarray<const T>(Shape(this->kernel_conf().reduce_sum_conf().kept_dims_shape()),
                             tmp_blob->dptr<T>()));

  const int64_t num_axes = dx_blob->shape().NumAxes();
  if (this->op_conf().reduce_mean_grad_conf().has_kept_dims_shape()) {
    NdarrayUtil<device_type, T>::BroadcastTo(
        ctx.device_ctx, XpuVarNdarray<T>(dx_blob, num_axes),
        XpuVarNdarray<const T>(Shape(this->op_conf().reduce_mean_grad_conf().kept_dims_shape()),
                               tmp_blob->dptr<T>()));
  } else {
    NdarrayUtil<device_type, T>::BroadcastTo(ctx.device_ctx, XpuVarNdarray<T>(dx_blob, num_axes),
                                             XpuVarNdarray<const T>(tmp_blob, num_axes));
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceMeanGradConf, ReduceMeanGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
