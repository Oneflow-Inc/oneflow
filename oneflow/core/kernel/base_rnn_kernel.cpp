#include "oneflow/core/kernel/base_rnn_kernel.h"

namespace oneflow {

namespace {

Blob* GetHiddenBlob(std::function<Blob*(std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("h0"); }
  return BnInOp2Blob("ht_1");
}

bool Ish0Model(const KernelConf& kernel_conf) {
  auto& input_bns = kernel_conf.input_bns();
  return find(input_bns.begin(), input_bns.end(), "h0") == input_bns.end();
}

}  // namespace

template<DeviceType device_type, typename T>
void BaseRnnKernel<device_type, T>::ForwardDataContent(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* in_ip_op_weight_blob = BnInOp2Blob("in_ip_op_weight");
  Blob* in_ip_op_out_blob = BnInOp2Blob("in_ip_op_out");
  const Blob* ht_1_blob = GetHiddenBlob();
  const Blob* hidden_ip_op_weight_blob = BnInOp2Blob("hidden_ip_op_weight");
  Blob* hidden_ip_op_out_blob = BnInOp2Blob("hidden_ip_op_out");
  Blob* plus_op_out_blob = BnInOp2Blob("plus_op_out");
  Blob* f_op_out_blob = BnInOp2Blob("f_op_out");
  Blob* ht_blob = BnInOp2Blob("ht");
  Blob* out_blob = BnInOp2Blob("out");

  // in_ip_op_out = in * in_ip_op_weight
  BlasMatrixMatrix<device_type, T>(
      ctx, CblasNoTrans, CblasTrans, static_cast<T>(1), static_cast<T>(0),
      in_blob, in_ip_op_weight_blob, in_ip_op_out_blob);
  // hidden_ip_op_out = ht_1 * hidden_ip_op_weight
  BlasMatrixMatrix<device_type, T>(
      ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1), static_cast<T>(0),
      ht_1_blob, hidden_ip_op_weight_blob, hidden_ip_op_out_blob);
  // plus_op_out = in_ip_op_out + hidden_ip_op_out
  BaseRnnKernelUtil<device_type, T>::Add(
      ctx, plus_op_out_blob->shape().elem_cnt();
      in_ip_op_out_blob->dptr<T>(), hidden_ip_op_out_blob->dptr<T>(),
      plus_op_out_blob->mut_dptr<T>());
  if (this->op_conf().recurrent_conf().has_bias_term()) {
    const Blob* bias_blob = BnInOp2Blob("bias");
    const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
    // plus_op_out += bias * bias_multiplier
    BlasMatrixMatrix<device_type, T>(
        ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1), static_cast<T>(1),
        bias_mul_blob, bias_blob, plus_op_out_blob);
  }
  // f_op_out = tanh(plus_op_out)
  BaseRnnKernelUtil<device_type, T>::Tanh(
      ctx, f_op_out_blob->shape()->elem_cnt(), plus_op_out_blob->dptr<T>(),
      f_op_out_blob->mut_dptr<T>());
  // ht = f_op_out
  KernelUtil<device_type>::Memcpy(ctx, ht_blob->mut_dptr<T>(),
                                  f_op_out_blob->dptr<T>());
  // out = f_op_out
  KernelUtil<device_type>::Memcpy(ctx, out_blob->mut_dptr<T>(),
                                  f_op_out_blob->dptr<T>());
}

template<DeviceType device_type, typename T>
void BaseRnnKernel<DeviceType::kCPU, T>::ForwardDataId(
    const KernelCtx&,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
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

  Blob* plus_op_out_diff_blob = plus_op_out_diff_blob;
  if (recurrent_conf().has_bias_term()) {
    plus_op_out_diff_blob = BnInOp2Blob("bias_diff_blob");
  }
  Blob* diff_sum_blob = BnInOp2Blob("f_op_out");
  // diff_sum == ht_diff + out_diff
  BaseRnnKernelUtil<device_type, T>::Add(ctx, ht_diff_blob->dptr<T>(),
                                         out_diff_blob->dptr<T>(),
                                         diff_sum_blob->mut_dptr<T>());
  // plus_op_out_diff = (1 - ht^2) .* ht_diff
  BaseRnnKernelUtil<device_type, T>::ComputePlusOutDiff(
      ctx, ht_blob->dptr<t>(), diff_sum_blob->dptr<T>(),
      plus_op_out_diff_blob->mut_dptr<T>());
  // hidden_ip_op_weight_diff = plus_op_out_diff * ht_1
  BlasMatrixMatrix<device_type, T>(
      ctx, CblasNoTrans, CblasTrans, static_cast<T>(1), static_cast<T>(0),
      plus_op_out_diff_blob, ht_1_blob, hidden_ip_op_weight_diff_blob);
  // in_ip_op_weight_diff = plus_op_out_diff * in
  BlasMatrixMatrix<device_type, T>(
      ctx, CblasNoTrans, CblasTrans, static_cast<T>(1), static_cast<T>(0),
      plus_op_out_diff_blob, in_blob, in_ip_op_weight_diff_blob);
  // in_diff = plus_op_out_diff * in_ip_op_weight_diff
  BlasMatrixMatrix<device_type, T>(
      ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1), static_cast<T>(0),
      plus_op_out_diff_blob, in_ip_op_weight_diff_blob, in_diff_blob);
  if (Ish0Model(this->kernel_conf()) && out_diff_blob->col_id() == 0) {
    h0_diff_blob = BnInOp2Blob("h0_diff");
    BlasMatrixMatrix<device_type, T>(
        ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(0), static_cast<T>(0),
        plus_op_out_diff_blob, hidden_ip_op_weight_diff, h0_diff_blob);
  }
}

template<DeviceType device_type, typename T>
void BaseRnnKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    const KernelCtx&, std::mt19937,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(this->op_conf().recurrent_conf(),
                        in_ip_op_weight_initializer),
      random_seed_gen(), BnInOp2Blob("in_ip_op_weight"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(this->op_conf().recurrent_conf(),
                        hidden_ip_op_weight_initializer),
      random_seed_gen(), BnInOp2Blob("hidden_ip_op_weight"));
  if (this->op_conf().recurrent_conf().has_bias_term()) {
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx.device_ctx,
        OF_PB_POINTER_GET(this->op_conf().recurrent_conf(), bias_initializer),
        random_seed_gen(), BnInOp2Blob("bias"));
  }
  if (IsH0Model(this->kernel_conf())) {
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx.device_ctx,
        OF_PB_POINTER_GET(this->op_conf().recurrent_conf(),
                          init_hidden_initializer),
        random_seed_gen(), BnInOp2Blob("h0"));
  }
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
  if (this->op_conf().recurrent_conf().has_bias_term()) {
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
    auto sigmoid = [](T x) {
      return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    } FOR_RANGE(int64_t, i, 0, n) {
      y[i] = static_cast<T>(2) * sigmoid(2 * x[i]) - static_cast<T>(1);
    }
  }
  static void ComputePlusOutDiff(DeviceCtx* ctx, int64_t n, const T* ht,
                                 const T* ht_diff, T* plus_out_diff) {
    FOR_RANGE(int64_t, i, 0, n) {
      plus_out_diff[i] = (1 - ht[i] * ht[i]) * ht_diff[i];
    }
  }
};

}  // namespace oneflow
