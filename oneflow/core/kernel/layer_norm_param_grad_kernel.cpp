#include "oneflow/core/kernel/layer_norm_param_grad_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void LayerNormParamGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const LayerNormParamGradOpConf& conf = this->op_conf().layer_norm_param_grad_conf();
  const Blob* dy = BnInOp2Blob("dy");
  if (conf.has_beta_diff()) {
    Blob* reduce_buf = BnInOp2Blob("reduce_buf");
    Blob* beta_diff = BnInOp2Blob("beta_diff");
    const int64_t m = beta_diff->shape().elem_cnt();
    CHECK_EQ(dy->shape().elem_cnt() % m, 0);
    const int64_t n = dy->shape().elem_cnt() / m;
    NdarrayUtil<device_type, T>::ReduceSum(ctx.device_ctx,
                                           XpuVarNdarray<T>({1, m}, beta_diff->mut_dptr<T>()),
                                           XpuVarNdarray<const T>({n, m}, dy->dptr<T>()),
                                           XpuVarNdarray<T>({n, m}, reduce_buf->mut_dptr<T>()));
  }
  if (conf.has_gamma_diff()) {
    const Blob* normalized = BnInOp2Blob("normalized");
    Blob* reduce_buf = BnInOp2Blob("reduce_buf");
    Blob* gamma_diff = BnInOp2Blob("gamma_diff");
    const int64_t m = gamma_diff->shape().elem_cnt();
    CHECK_EQ(dy->shape().elem_cnt() % m, 0);
    const int64_t n = dy->shape().elem_cnt() / m;
    NdarrayUtil<device_type, T>::BroadcastMul(ctx.device_ctx,
                                              XpuVarNdarray<T>({n, m}, reduce_buf->mut_dptr<T>()),
                                              XpuVarNdarray<const T>({n, m}, normalized->dptr<T>()),
                                              XpuVarNdarray<const T>({n, m}, dy->dptr<T>()));
    NdarrayUtil<device_type, T>::ReduceSum(ctx.device_ctx,
                                           XpuVarNdarray<T>({1, m}, gamma_diff->mut_dptr<T>()),
                                           XpuVarNdarray<const T>({n, m}, reduce_buf->dptr<T>()),
                                           XpuVarNdarray<T>({n, m}, reduce_buf->mut_dptr<T>()));
  }
  if (conf.has_normalized_diff()) {
    Blob* normalized_diff = BnInOp2Blob("normalized_diff");
    if (conf.has_gamma()) {
      const Blob* gamma = BnInOp2Blob("gamma");
      const int64_t m = gamma->shape().elem_cnt();
      CHECK_EQ(dy->shape().elem_cnt() % m, 0);
      const int64_t n = dy->shape().elem_cnt() / m;
      NdarrayUtil<device_type, T>::BroadcastMul(
          ctx.device_ctx, XpuVarNdarray<T>({n, m}, normalized_diff->mut_dptr<T>()),
          XpuVarNdarray<const T>({n, m}, dy->dptr<T>()),
          XpuVarNdarray<const T>({1, m}, gamma->dptr<T>()));
    } else {
      normalized_diff->CopyDataContentFrom(ctx.device_ctx, dy);
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLayerNormParamGradConf, LayerNormParamGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
