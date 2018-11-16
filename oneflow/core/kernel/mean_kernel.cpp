#include "oneflow/core/kernel/mean_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MeanKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* mean_mul_blob = BnInOp2Blob("mean_multiplier");
  Blob* out_blob = BnInOp2Blob("out");
  size_t mean_dim_size = in_blob->shape().dim_vec().back();

  // out = in * mean_mul
  const int k = mean_dim_size;
  const int m = in_blob->shape().elem_cnt() / k;
  const int n = 1;
  KernelUtil<device_type, T>::OFGemm(ctx.device_ctx, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                                     in_blob->dptr<T>(), mean_mul_blob->dptr<T>(), 0.0,
                                     out_blob->mut_dptr<T>());

  KernelUtil<device_type, T>::Div(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  out_blob->mut_dptr<T>(), static_cast<T>(mean_dim_size));
}

template<DeviceType device_type, typename T>
void MeanKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  CHECK_EQ(1, out_diff_blob->shape().dim_vec().back());
  const Blob* mean_mul_blob = BnInOp2Blob("mean_multiplier");
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  Blob* bw_tmp_blob = BnInOp2Blob("bw_tmp_blob");

  Memcpy<device_type>(ctx.device_ctx, bw_tmp_blob->mut_dptr(), out_diff_blob->dptr(),
                      out_diff_blob->ByteSizeOfDataContentField());
  size_t mean_dim_size = in_diff_blob->shape().dim_vec().back();
  // bw_tmp = out_diff/M
  KernelUtil<device_type, T>::Div(ctx.device_ctx, bw_tmp_blob->shape().elem_cnt(),
                                  bw_tmp_blob->mut_dptr<T>(), static_cast<T>(mean_dim_size));
  // in_diff = bw_tmp * mean_mul'
  const int k = 1;
  const int n = mean_dim_size;
  const int m = in_diff_blob->shape().elem_cnt() / n;
  KernelUtil<device_type, T>::OFGemm(ctx.device_ctx, CblasNoTrans, CblasTrans, m, n, k, 1.0,
                                     bw_tmp_blob->dptr<T>(), mean_mul_blob->dptr<T>(), 0.0,
                                     in_diff_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void MeanKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf mean_multiplier_initializer_conf;
  mean_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
  KernelUtil<device_type, T>::InitializeWithConf(ctx, mean_multiplier_initializer_conf, 0,
                                                 BnInOp2Blob("mean_multiplier"));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMeanConf, MeanKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
