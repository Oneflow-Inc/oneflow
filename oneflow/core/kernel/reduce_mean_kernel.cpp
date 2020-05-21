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
  const ReduceMeanOpConf& conf = this->op_conf().reduce_mean_conf();
  const Shape& reduced_shape =
      conf.axis().empty()
          ? Shape::Ones(in_blob->shape().NumAxes())
          : CreateReducedShape(in_blob->shape(), {conf.axis().begin(), conf.axis().end()});
  NdarrayUtil<device_type, T>::ReduceSum(
      ctx.device_ctx, XpuVarNdarray<T>(reduced_shape, out_blob->mut_dptr<T>()),
      XpuVarNdarray<const T>(in_blob, in_blob->shape().NumAxes()),
      XpuVarNdarray<T>(fw_tmp_blob, in_blob->shape().NumAxes()));
  KernelUtil<device_type, T>::Div(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  out_blob->mut_dptr<T>(), static_cast<T>(count));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceMeanConf, ReduceMeanKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
