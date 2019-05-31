#include "oneflow/core/kernel/basic_rnn_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  ActivationType activation_type = this->op_conf().basic_rnn_conf().activation();
  if (activation_type == kTanH) {
    activation_fw_func_ = &KernelUtil<device_type, T>::TanH;
    activation_bw_func_ = &BasicRnnKernelUtil<device_type, T>::ComputeTanHDiff;
    last_colnum_activation_bw_func_ = &KernelUtil<device_type, T>::TanHBackward;
  } else if (activation_type == kSigmoid) {
    activation_fw_func_ = &KernelUtil<device_type, T>::Sigmoid;
    activation_bw_func_ = &BasicRnnKernelUtil<device_type, T>::ComputeSigmoidDiff;
    last_colnum_activation_bw_func_ = &KernelUtil<device_type, T>::SigmoidBackward;
  } else if (activation_type == kRelu) {
    activation_fw_func_ = &KernelUtil<device_type, T>::Relu;
    activation_bw_func_ = &BasicRnnKernelUtil<device_type, T>::ComputeReluDiff;
    last_colnum_activation_bw_func_ = &KernelUtil<device_type, T>::ReluBackward;
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
const PbMessage& BasicRnnKernel<device_type, T>::GetRecurrentOpConf() const {
  return this->op_conf().basic_rnn_conf();
}

template<DeviceType device_type, typename T>
bool BasicRnnKernel<device_type, T>::HasInitHiddenInitializer() const {
  return this->op_conf().basic_rnn_conf().has_init_hidden_initializer();
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* plus_op_out_blob = BnInOp2Blob("plus_op_out");
  Blob* out_blob = BnInOp2Blob("out");

  // plus_op_out = in * i2h_weight
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans, OneVal<T>::value,
                                       ZeroVal<T>::value, BnInOp2Blob("in"),
                                       BnInOp2Blob("i2h_weight"), plus_op_out_blob);

  // plus_op_out += hidden * h2h_weight
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans, OneVal<T>::value,
                                       OneVal<T>::value, hidden_blob, BnInOp2Blob("h2h_weight"),
                                       plus_op_out_blob);

  // plus_op_out += bias_multiplier * bias
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasNoTrans, OneVal<T>::value,
                                       OneVal<T>::value, BnInOp2Blob("bias_multiplier"),
                                       BnInOp2Blob("bias"), plus_op_out_blob);

  // out = activation(plus_op_out)
  (*activation_fw_func_)(ctx.device_ctx, out_blob->shape().elem_cnt(), plus_op_out_blob->dptr<T>(),
                         out_blob->mut_dptr<T>());

  // rec_out = out
  BnInOp2Blob("rec_out")->CopyDataContentFrom(ctx.device_ctx, out_blob);
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  // reuse memory
  const Blob* plus_op_out_blob = BnInOp2Blob("plus_op_out");
  Blob* plus_op_out_diff_blob = BnInOp2Blob("plus_op_out");

  if (in_blob->col_id() == in_blob->max_col_id()) {
    (*last_colnum_activation_bw_func_)(
        ctx.device_ctx, out_blob->shape().elem_cnt(), plus_op_out_blob->dptr<T>(),
        out_blob->dptr<T>(), out_diff_blob->dptr<T>(), plus_op_out_diff_blob->mut_dptr<T>());
  } else {
    (*activation_bw_func_)(ctx.device_ctx, out_blob->shape().elem_cnt(), out_blob->dptr<T>(),
                           out_diff_blob->dptr<T>(), BnInOp2Blob("rec_out_diff")->dptr<T>(),
                           plus_op_out_diff_blob->mut_dptr<T>());
  }

  if (this->op_conf().trainable()) {
    // h2h_weight_diff = plus_op_out_diff * hidden
    KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans, OneVal<T>::value,
                                         ZeroVal<T>::value, plus_op_out_diff_blob, hidden_blob,
                                         BnInOp2Blob("h2h_weight_diff"));

    // i2h_weight_diff = plus_op_out_diff * in
    KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans, OneVal<T>::value,
                                         ZeroVal<T>::value, plus_op_out_diff_blob,
                                         BnInOp2Blob("in"), BnInOp2Blob("i2h_weight_diff"));
  }

  if (BnInOp2Blob("in_diff") != nullptr) {
    // in_diff = plus_op_out_diff * i2h_weight
    KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasNoTrans,
                                         OneVal<T>::value, ZeroVal<T>::value, plus_op_out_diff_blob,
                                         BnInOp2Blob("i2h_weight"), BnInOp2Blob("in_diff"));
  }

  if (BnInOp2Blob("bias_diff") != nullptr && this->op_conf().trainable()) {
    // bias_diff = bias_multiplier * plus_op_out_diff
    KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans, OneVal<T>::value,
                                         ZeroVal<T>::value, BnInOp2Blob("bias_multiplier"),
                                         plus_op_out_diff_blob, BnInOp2Blob("bias_diff"));
  }

  if (BnInOp2Blob("in")->col_id() != 0 || this->NeedExternalH0()
      || this->op_conf().basic_rnn_conf().is_init_hidden_trainable()) {
    // hidden_diff = plus_op_out_diff * h2h_weight
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, OneVal<T>::value, ZeroVal<T>::value,
        plus_op_out_diff_blob, BnInOp2Blob("h2h_weight"), this->GetHiddenDiffBlob(BnInOp2Blob));
  }
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf bias_multiplier_fill_conf;
  bias_multiplier_fill_conf.mutable_constant_conf()->set_value(1.f);
  KernelUtil<device_type, T>::InitializeWithConf(ctx, bias_multiplier_fill_conf, 0,
                                                 BnInOp2Blob("bias_multiplier"));
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::VirtualInitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      this->GetInitializerFromPbMessage(this->op_conf().basic_rnn_conf(), "i2h_weight_initializer"),
      (*random_seed_gen)(), BnInOp2Blob("i2h_weight"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      this->GetInitializerFromPbMessage(this->op_conf().basic_rnn_conf(), "h2h_weight_initializer"),
      (*random_seed_gen)(), BnInOp2Blob("h2h_weight"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx, this->GetInitializerFromPbMessage(this->op_conf().basic_rnn_conf(), "bias_initializer"),
      (*random_seed_gen)(), BnInOp2Blob("bias"));
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::VirtualInitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* i2h_weight_blob = BnInOp2Blob("i2h_weight");
  KernelUtil<device_type, T>::InitializeWithDir(
      ctx, part_id, part_num, model_load_dir, i2h_weight_blob, "i2h_weight",
      i2h_weight_blob->shape().At(0), i2h_weight_blob->shape().Count(1));
  Blob* h2h_weight_blob = BnInOp2Blob("h2h_weight");
  KernelUtil<device_type, T>::InitializeWithDir(
      ctx, part_id, part_num, model_load_dir, h2h_weight_blob, "h2h_weight",
      h2h_weight_blob->shape().At(0), h2h_weight_blob->shape().Count(1));
  KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir,
                                                BnInOp2Blob("bias"), "bias",
                                                BnInOp2Blob("bias")->shape().At(0), 1);
}

template<typename T>
struct BasicRnnKernelUtil<DeviceType::kCPU, T> {
  static void ComputeTanHDiff(DeviceCtx* ctx, int64_t n, const T* out, const T* out_diff,
                              const T* rec_out_diff, T* plus_out_diff) {
    FOR_RANGE(int64_t, i, 0, n) {
      plus_out_diff[i] = (1 - out[i] * out[i]) * (out_diff[i] + rec_out_diff[i]);
    }
  }
  static void ComputeSigmoidDiff(DeviceCtx* ctx, int64_t n, const T* out, const T* out_diff,
                                 const T* rec_out_diff, T* plus_out_diff) {
    FOR_RANGE(int64_t, i, 0, n) {
      plus_out_diff[i] = out[i] * (1 - out[i]) * (out_diff[i] + rec_out_diff[i]);
    }
  }
  static void ComputeReluDiff(DeviceCtx* ctx, int64_t n, const T* out, const T* out_diff,
                              const T* rec_out_diff, T* plus_out_diff) {
    FOR_RANGE(int64_t, i, 0, n) { plus_out_diff[i] = out[i] * (out_diff[i] + rec_out_diff[i]); }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBasicRnnConf, BasicRnnKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
