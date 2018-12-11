#include "oneflow/core/kernel/reduce_mean_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceMeanKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* fw_tmp_blob = BnInOp2Blob("fw_tmp");
  size_t count = in_blob->shape().elem_cnt() / out_blob->shape().elem_cnt();
  NdarrayUtil<device_type, T>::ReduceSum(
      ctx.device_ctx,
      XpuVarNdarray<T>(Shape(this->kernel_conf().reduce_sum_conf().kept_dims_shape()),
                       out_blob->mut_dptr<T>()),
      XpuVarNdarray<const T>(in_blob, in_blob->shape().NumAxes()),
      XpuVarNdarray<T>(fw_tmp_blob, in_blob->shape().NumAxes()));
  KernelUtil<device_type, T>::Div(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  out_blob->mut_dptr<T>(), static_cast<T>(count));
}

template<DeviceType device_type, typename T>
void ReduceMeanKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  Blob* bw_tmp_blob = BnInOp2Blob("bw_tmp");
  size_t count = in_diff_blob->shape().elem_cnt() / out_diff_blob->shape().elem_cnt();
  Memcpy<device_type>(ctx.device_ctx, bw_tmp_blob->mut_dptr(), out_diff_blob->dptr(),
                      out_diff_blob->ByteSizeOfDataContentField());
  KernelUtil<device_type, T>::Div(ctx.device_ctx, bw_tmp_blob->shape().elem_cnt(),
                                  bw_tmp_blob->mut_dptr<T>(), static_cast<T>(count));
  NdarrayUtil<device_type, T>::BroadcastTo(
      ctx.device_ctx, XpuVarNdarray<T>(in_diff_blob, in_diff_blob->shape().NumAxes()),
      XpuVarNdarray<const T>(Shape(this->kernel_conf().reduce_sum_conf().kept_dims_shape()),
                             bw_tmp_blob->dptr<T>()));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceMeanConf, ReduceMeanKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
