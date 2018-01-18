#include "oneflow/core/kernel/basic_rnn_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_ip_op_out_blob = BnInOp2Blob("in_ip_op_out");
  const Blob* ht_1_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* hidden_ip_op_out_blob = BnInOp2Blob("hidden_ip_op_out");
  Blob* plus_op_out_blob = BnInOp2Blob("plus_op_out");
  Blob* ht_blob = BnInOp2Blob("ht");

  // plus_op_out = in * in_ip_op_weight
  KernelUtil<device_type, T>::BlasMatrixMatrix(
      ctx, CblasNoTrans, CblasTrans, static_cast<T>(1), static_cast<T>(0),
      BnInOp2Blob("in"), BnInOp2Blob("in_ip_op_weight"), plus_op_out_blob);

  // plus_ip_op_out += ht_1 * hidden_ip_op_weight
  KernelUtil<device_type, T>::BlasMatrixMatrix(
      ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1), static_cast<T>(1),
      ht_1_blob, BnInOp2Blob("hidden_ip_op_weight"), plus_op_out_blob);

  if (this->op_conf().recurrent_conf().has_bias_term()) {
    const Blob* bias_blob = BnInOp2Blob("bias");
    const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
    // plus_op_out += bias * bias_multiplier
    KernelUtil<device_type, T>::BlasMatrixMatrix(
        ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1), static_cast<T>(1),
        bias_mul_blob, bias_blob, plus_op_out_blob);
  }
  // ht = tanh(plus_op_out)
  BasicRnnKernelUtil<device_type, T>::Tanh(
      ctx.device_ctx, ht_blob->shape().elem_cnt(), plus_op_out_blob->dptr<T>(),
      ht_blob->mut_dptr<T>());
  // rec_ht = ht
  BnInOp2Blob("rec_ht")->CopyDataContentFrom<device_type>(ctx.device_ctx,
                                                          ht_blob);
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("rec_ht")->CopyDataIdFrom<device_type>(ctx.device_ctx,
                                                     BnInOp2Blob("in"));
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* ht_diff_blob = BnInOp2Blob("ht_diff");
  const Blob* ht_blob = BnInOp2Blob("ht");
  const Blob* ht_1_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* in_ip_op_weight_diff_blob = BnInOp2Blob("in_ip_op_weight_diff");
  // reuse memory
  Blob* diff_sum_blob = BnInOp2Blob("plus_op_out");
  Blob* plus_op_out_diff_blob = BnInOp2Blob("plus_op_out");

  // diff_sum = ht_diff + rec_ht_diff
  BasicRnnKernelUtil<device_type, T>::Add(
      ctx.device_ctx, ht_diff_blob->shape().elem_cnt(), ht_diff_blob->dptr<T>(),
      BnInOp2Blob("rec_ht_diff")->dptr<T>(), diff_sum_blob->mut_dptr<T>());

  // plus_op_out_diff = (1 - ht^2) .* diff_sum
  BasicRnnKernelUtil<device_type, T>::ComputePlusOutDiff(
      ctx.device_ctx, ht_blob->shape().elem_cnt(), ht_blob->dptr<T>(),
      diff_sum_blob->dptr<T>(), plus_op_out_diff_blob->mut_dptr<T>());

  // hidden_ip_op_weight_diff = plus_op_out_diff * ht_1
  KernelUtil<device_type, T>::BlasMatrixMatrix(
      ctx, CblasNoTrans, CblasTrans, static_cast<T>(1), static_cast<T>(0),
      plus_op_out_diff_blob, ht_1_blob,
      BnInOp2Blob("hidden_ip_op_weight_diff"));

  // in_ip_op_weight_diff = plus_op_out_diff * in
  KernelUtil<device_type, T>::BlasMatrixMatrix(
      ctx, CblasNoTrans, CblasTrans, static_cast<T>(1), static_cast<T>(0),
      plus_op_out_diff_blob, BnInOp2Blob("in"), in_ip_op_weight_diff_blob);

  // in_diff = plus_op_out_diff * in_ip_op_weight_diff
  KernelUtil<device_type, T>::BlasMatrixMatrix(
      ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1), static_cast<T>(0),
      plus_op_out_diff_blob, in_ip_op_weight_diff_blob, BnInOp2Blob("in_diff"));

  if (this->op_conf().recurrent_conf().has_bias_term()) {
    // bias_diff = plus_op_out_diff * bias_multiplier
    KernelUtil<device_type, T>::BlasMatrixMatrix(
        ctx, CblasTrans, CblasNoTrans, static_cast<T>(1), static_cast<T>(0),
        BnInOp2Blob("bias_multiplier"), plus_op_out_diff_blob,
        BnInOp2Blob("bias_diff"));
  }
  if (this->Ish0Model() && BnInOp2Blob("rec_ht_diff")->col_id() == 0) {
    // ho_diff = plus_op_out_diff * hidden_ip_op_weight
    KernelUtil<device_type, T>::BlasMatrixMatrix(
        ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(0), static_cast<T>(0),
        plus_op_out_diff_blob, BnInOp2Blob("hidden_ip_op_weight"),
        BnInOp2Blob("h0_diff"));
  }
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::VirtualInitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(this->op_conf().recurrent_conf().basic_rnn_cell(),
                        in_ip_op_weight_initializer),
      random_seed_gen(), BnInOp2Blob("in_ip_op_weight"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(this->op_conf().recurrent_conf().basic_rnn_cell(),
                        hidden_ip_op_weight_initializer),
      random_seed_gen(), BnInOp2Blob("hidden_ip_op_weight"));
  if (this->op_conf().recurrent_conf().has_bias_term()) {
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx.device_ctx,
        OF_PB_POINTER_GET(this->op_conf().recurrent_conf().basic_rnn_cell(),
                          bias_initializer),
        random_seed_gen(), BnInOp2Blob("bias"));
  }
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::VirtualInitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_ip_op_weight_blob = BnInOp2Blob("in_ip_op_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir, in_ip_op_weight_blob,
      "in_ip_op_weight", in_ip_op_weight_blob->shape().At(0),
      in_ip_op_weight_blob->shape().Count(1));
  Blob* hidden_ip_op_weight_blob = BnInOp2Blob("hidden_ip_op_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir,
      hidden_ip_op_weight_blob, "hidden_ip_op_weight",
      hidden_ip_op_weight_blob->shape().At(0),
      hidden_ip_op_weight_blob->shape().Count(1));
  if (this->op_conf().recurrent_conf().has_bias_term()) {
    KernelUtil<device_type, T>::InitializeWithModelDir(
        ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias"),
        "bias", BnInOp2Blob("bias")->shape().At(0), 1);
  }
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::InitModelTmpBlobs(
    const KernelCtx& ctx, const ParallelContext* parallel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->op_conf().recurrent_conf().has_bias_term()) {
    InitializerConf bias_multiplier_fill_conf;
    bias_multiplier_fill_conf.mutable_constant_conf()->set_value(1.f);
    KernelUtil<device_type, T>::Initialize(ctx.device_ctx,
                                           bias_multiplier_fill_conf, 0,
                                           BnInOp2Blob("bias_multiplier"));
  }
}

template<typename T>
class BasicRnnKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void Add(DeviceCtx* ctx, int64_t n, const T* x, const T* y, T* z) {
    FOR_RANGE(int64_t, i, 0, n) { z[i] = x[i] + y[i]; }
  }
  static void Tanh(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
    auto sigmoid = [](T x) {
      return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    };
    FOR_RANGE(int64_t, i, 0, n) {
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

#define INSTANTIATE_KERNEL(device_type, data_type_pair) \
  template class BasicRnnKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ)

template class BasicRnnKernelUtil<DeviceType::kCPU, float>;
template class BasicRnnKernelUtil<DeviceType::kCPU, double>;

}  // namespace oneflow
