#include "oneflow/core/kernel/layer_norm_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

InitializerConf ConstantInitializerConf(float val) {
  InitializerConf conf;
  conf.mutable_constant_conf()->set_value(val);
  return conf;
}

InitializerConf OnesInitializerConf() { return ConstantInitializerConf(1.0f); }

InitializerConf ZerosInitializerConf() { return ConstantInitializerConf(0.0f); }

}  // namespace

template<DeviceType device_type, typename T>
void LayerNormConstBufInitUtil<device_type, T>::InitConstBufBlobsImpl(
    DeviceCtx* ctx, const InitializerConf& initializer_conf, uint32_t random_seed, Blob* blob) {
  KernelUtil<device_type, T>::InitializeWithConf(ctx, initializer_conf, 0, blob);
}

template<DeviceType device_type>
void LayerNormConstBufInitUtil<device_type, float16>::InitConstBufBlobsImpl(
    DeviceCtx* ctx, const InitializerConf& initializer_conf, uint32_t random_seed, Blob* blob) {
  KernelUtil<device_type, float>::InitializeWithConf(ctx, initializer_conf, 0, blob);
}

template<DeviceType device_type, typename T>
void LayerNormKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const LayerNormOpConf& conf = this->op_conf().layer_norm_conf();
  const Blob* in = BnInOp2Blob("in");
  const Blob* bn_scale = BnInOp2Blob("cudnn_bn_scale_ones");
  const Blob* bn_bias = BnInOp2Blob("cudnn_bn_bias_zeros");
  Blob* out = BnInOp2Blob("out");
  Blob* normalize_out = conf.scale() ? BnInOp2Blob("normalized") : out;
  Blob* mean = BnInOp2Blob("mean");
  Blob* inv_variance = BnInOp2Blob("inv_variance");
  LayerNormKernelUtil<device_type, T>::NormalizeForward(
      ctx.device_ctx, in, bn_scale, bn_bias, conf.epsilon(), normalize_out, mean, inv_variance);
  if (conf.scale()) {
    const Blob* gamma = BnInOp2Blob("gamma");
    const int64_t m = gamma->shape().elem_cnt();
    CHECK_EQ(out->shape().elem_cnt() % m, 0);
    const int64_t n = out->shape().elem_cnt() / m;
    NdarrayUtil<device_type, T>::BroadcastMul(
        ctx.device_ctx, XpuVarNdarray<T>({n, m}, out->mut_dptr<T>()),
        XpuVarNdarray<const T>({n, m}, normalize_out->dptr<T>()),
        XpuVarNdarray<const T>({1, m}, gamma->dptr<T>()));
  }
  if (conf.center()) {
    const Blob* beta = BnInOp2Blob("beta");
    const int64_t m = beta->shape().elem_cnt();
    CHECK_EQ(out->shape().elem_cnt() % m, 0);
    const int64_t n = out->shape().elem_cnt() / m;
    NdarrayUtil<device_type, T>::BroadcastAdd(ctx.device_ctx,
                                              XpuVarNdarray<T>({n, m}, out->mut_dptr<T>()),
                                              XpuVarNdarray<const T>({n, m}, out->dptr<T>()),
                                              XpuVarNdarray<const T>({1, m}, beta->dptr<T>()));
  }
}

template<DeviceType device_type, typename T>
void LayerNormKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf ones_initializer = OnesInitializerConf();
  LayerNormConstBufInitUtil<device_type, T>::InitConstBufBlobsImpl(
      ctx, ones_initializer, 0, BnInOp2Blob("cudnn_bn_scale_ones"));
  InitializerConf zeros_initializer = ZerosInitializerConf();
  LayerNormConstBufInitUtil<device_type, T>::InitConstBufBlobsImpl(
      ctx, zeros_initializer, 0, BnInOp2Blob("cudnn_bn_bias_zeros"));
}

template<typename T>
struct LayerNormKernelUtil<DeviceType::kCPU, T> {
  static void NormalizeForward(const DeviceCtx* ctx, const Blob* in, const Blob* scale,
                               const Blob* bias, double epsilon, Blob* out, Blob* mean,
                               Blob* inv_variance) {
    UNIMPLEMENTED();
  }
  static void NormalizeBackward(const DeviceCtx* ctx, const Blob* in, const Blob* scale,
                                const Blob* mean, const Blob* inv_variance, const Blob* out_diff,
                                double epsilon, Blob* in_diff, Blob* scale_diff, Blob* bias_diff) {
    UNIMPLEMENTED();
  }
};

#define INSTANTIATE_LAYER_NORM_KERNEL_UTIL_CPU(type_cpp, type_proto) \
  template struct LayerNormKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_LAYER_NORM_KERNEL_UTIL_CPU, FLOATING_DATA_TYPE_SEQ)
#undef INSTANTIATE_LAYER_NORM_KERNEL_UTIL_CPU

#define INSTANTIATE_LAYER_NORM_CONST_BUF_INIT_UTIL(device_type, data_type_pair) \
  template class LayerNormConstBufInitUtil<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_LAYER_NORM_CONST_BUF_INIT_UTIL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ);
template struct LayerNormConstBufInitUtil<DeviceType::kGPU, float16>;

ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(OperatorConf::kLayerNormConf, LayerNormKernel,
                                         FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
