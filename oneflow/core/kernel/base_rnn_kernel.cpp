#include "oneflow/core/kernel/base_rnn_kernel.h"

namespace oneflow {

namespace {

Blob* GetHiddenBlob(std::function<> BnInOp2Blob) const {
  // TODO
  return nullptr;
}

} // namespace

template<DeviceType device_type, typename T>
void BaseRnnKernel<device_type, T>::ForwardDataContent(
    const KernelCtx&, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* in_ip_op_weight_blob = BnInOp2Blob("in_ip_op_weight");
  Blob* in_ip_op_out_blob = BnInOp2Blob("in_ip_op_out");
  const Blob* ht_1_blob = GetHiddenBlob();
  const Blob* hidden_ip_op_weight_blob = BnInOp2Blob("hidden_ip_op_weight");
  Blob* hidden_ip_op_out_blob = BnInOp2Blob("hidden_ip_op_out");
  Blob* plus_op_out_blob = BnInOp2Blob("plus_op_out");
  Blob* f_op_out_blob = BnInOp2Blob("f_op_out");
  Blob* ht_blob = BnInOp2Blob("ht");

  BlasMatrixMatrix<device_type, T>(ctx, CblasNoTrans, CblasNoTrans,
                                   static_cast<T>(1.0), static_cast<T>(0.0),
                                   in_blob, in_ip_op_weight_blob, in_ip_op_out_blob);
  BlasMatrixMatrix<device_type, T>(ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1.0),
                                   static_cast<T>(0.0), ht_1_blob,
                                   hidden_ip_op_weight_blob, hidden_ip_op_out_blob);
  BaseRnnKernelUtil<device_type, T>::Add(ctx, plus_op_out_blob->shape().elem_cnt();
                                         in_ip_op_out_blob->dptr<T>(),
                                         hidden_ip_op_out_blob->dptr<T>(),
                                         plus_op_out_blob->mut_dptr<T>());
  if (this->op_conf().base_rnn_conf().has_bias_term()) {
    const Blob* bias_blob = BnInOp2Blob("bias");
    const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
    BlasMatrixMatrix<device_type, T>(ctx, CblasNoTrans, CblasNoTrans,
                                     static_cast<T>(1.0), static_cast<T>(1.0),
                                     bias_mul_blob, bias_blob, plus_op_out_blob);
  }
  BaseRnnKernelUtil<device_type, T>::Tanh(ctx, f_op_out_blob->shape()->elem_cnt(),
                                          plus_op_out_blob->dptr<T>(),
                                          f_op_out_blob->mut_dptr<T>());
  KernelUtil<device_type>::Memcpy(ctx, ht_blob->mut_dptr<T>(), f_op_out_blob->dptr<T>());
}

template<DeviceType device_type, typename T>
void BaseRnnKernel<DeviceType::kCPU, T>::ForwardDataId(
    const KernelCtx&, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("ht")->CopyDataIdFrom(BnInOp2Blob("in"));
}

template<DeviceType device_type, typename T>
void BaseRnnKernel<device_type, T>::BackwardDataContent(
    const KernelCtx&, std::function<Blob*(const std::string&)>) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  Blob* ht_diff_blob = BnInOp2Blob("ht_diff");
  const Blob* ht_blob = BnInOp2Blob("ht");
  const Blob* ht_1_blob = GetHiddenBlob();
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* in_ip_op_weight_diff_blob = BnInOp2Blob("in_ip_op_weight_diff");
  Blob* hidden_ip_op_weight_diff_blob = BnInOp2Blob("hidden_ip_op_weight_diff");
  Blob* f_op_out_blob = BnInOp2Blob("f_op_out_blob");


}

template<DeviceType device_type, typename T>
void BaseRnnKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    const KernelCtx&, std::mt19937,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<DeviceType device_type, typename T>
void BaseRnnKernel<device_type, T>::InitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<DeviceType device_type, typename T>
void BaseRnnKernel<device_type, T>::InitModelTmpBlobs(
    const KernelCtx& ctx, const ParallelContext* parallel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->op_conf().base_rnn_conf().has_bias_term()) {
    FillConf bias_multiplier_fill_conf;
    bias_multiplier_fill_conf.mutable_constant_conf()->set_value(1.f);
    KernelUtil<device_type, T>::Fill(ctx.device_ctx, bias_multiplier_fill_conf,
                                     0, BnInOp2Blob("bias_multiplier"));
  }
}

template<typename T>
class BaseRnnKernel<DeviceType::kCPU, T> final {
 public:
  static void Add(DeviceCtx* ctx, int64_t n, const T* x, const T* y, T* z) {
    FOR_RANGE(int64_t, i, 0, n) { z[i] = x[i] + y[i]; }
  }
  static void Tanh(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
    FOR_RANGE(int64_t, i, 0, n) {
      y[i] = static_cast<T>(2) * sigmoid(2 * x[i]) - static_cast<T>(1);
    }
  }
};

}  // namespace oneflow
