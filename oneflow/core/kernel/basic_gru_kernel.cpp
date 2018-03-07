#include "oneflow/core/kernel/basic_gru_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& BasicGruKernel<device_type, T>::GetRecurrentOpConf() const {
  return this->op_conf().basic_gru_conf();
}

template<DeviceType device_type, typename T>
bool BasicGruKernel<device_type, T>::HasInitHiddenInitializer() const {
  return this->op_conf().basic_gru_conf().has_init_hidden_initializer();
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* plus_op_out_blob = BnInOp2Blob("plus_op_out");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* reset_gate_data_blob = BnInOp2Blob("reset_gate_data");
  Blob* update_gate_data_blob = BnInOp2Blob("update_gate_data");
  Blob* candidate_hidden_data_blob = BnInOp2Blob("candidate_hidden_data");
  Blob* reset_gate_out_blob = BnInOp2Blob("reset_gate_out");
  Blob* update_gate_out_blob = BnInOp2Blob("update_gate_out");
  Blob* candidate_hidden_out_blob = BnInOp2Blob("candidate_hidden_out");
  Blob* reset_mul_hidden_blob = BnInOp2Blob("reset_mul_hidden");
  Blob* reset_mul_candidate_hidden = BnInOp2Blob("reset_mul_candidate_hidden");

  // reset_gate_data = in * i2h_weight_r
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), BnInOp2Blob("in"), BnInOp2Blob("i2h_weight_r"),
      reset_gate_data_blob);
  // reset_gate_data += hidden * h2h_weight_r
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       hidden_blob, BnInOp2Blob("h2h_weight_r"),
                                       reset_gate_data_blob);
  // reset_gate_data += bias_multiplier_r * bias_r
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_multiplier_r"), BnInOp2Blob("bias_r"),
      reset_gate_data_blob);
  // reset_gate_out = sigmoid(reset_gate_data)
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, reset_gate_blob->shape().elem_cnt(),
      reset_gate_data_blob->dptr<T>(), reset_gate_out_blob->mut_dptr<T>());

  // update_gate_data = in * i2h_weight_z
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), BnInOp2Blob("in"), BnInOp2Blob("i2h_weight_z"),
      update_gate_data_blob);
  // update_gate_data += hidden * h2h_weight_z
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       hidden_blob, BnInOp2Blob("h2h_weight_z"),
                                       update_gate_data_blob);
  // update_gate_data += bias_multiplier_z * bias_z
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_multiplier_z"), BnInOp2Blob("bias_z"),
      update_gate_data_blob);
  // update_gate_out = sigmoid(update_gate_data)
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, update_gate_blob->shape().elem_cnt(),
      update_gate_data_blob->dptr<T>(), update_gate_out_blob->mut_dptr<T>());

  // candidate_hidden_data = hidden * h2h_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), hidden_blob, BnInOp2Blob("h2h_weight"),
      candidate_hidden_data_blob);

  // candidate_hidden_data *= reset_gate_out
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), candidate, reset_gate_out_blob,
      candidate_hidden_data_blob);

  // candidate_hidden_data += in * i2h_weight
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       BnInOp2Blob("in"), BnInOp2Blob("i2h_weight"),
                                       candidate_hidden_data_blob);

  // candidate_hidden_data += bias_multiplier * bias
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_multiplier"), BnInOp2Blob("bias"),
      candidate_hidden_data_blob);

  // candidate_hidden_out = tanh(candidate_hidden_data)
  KernelUtil<device_type, T>::TanH(
      ctx.device_ctx, update_gate_blob->shape().elem_cnt(),
      candidate_hidden_data_blob->dptr<T>(), candidate_hidden_out_blob->mut_dptr<T>());

  // plus_op_out = hidden .* reset_gate
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(hidden->shape().At(0)), reset_mul_hidden_blob, reset_gate_blob,
      plus_op_out_blob);
  // reset_mul_candidate_hidden = candidate_hidden .* reset_gate
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(hidden->shape().At(0)), reset_mul_candidate_hidden_blob, reset_gate_blob,
      reset_mul_candidate_hidden_blob);

  // plus_op_out -= reset_mul_candidate_hidden
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, static_cast<T>(hidden->shape().At(0)), static_cast<T>(-1),
      reset_mul_candidate_hidden_blob, static_cast<T>(1), plus_op_out_blob, static_cast<T>(1)};
  // plus_op_out += candidate_hidden
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, static_cast<T>(hidden->shape().At(0)), static_cast<T>(1),
      candidate_hidden_blob, static_cast<T>(1), plus_op_out_blob, static_cast<T>(1)};

  // rec_out = out
  BnInOp2Blob("rec_out")->CopyDataContentFrom(ctx.device_ctx, out_blob);
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_blob = BnInOp2Blob("out");
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* rec_out_diff_blob = BnInOp2Blob("rec_out_diff");
  // reuse memory
  Blob* plus_op_out_diff_blob = BnInOp2Blob("plus_op_out");

  if (this->op_conf().basic_rnn_conf().activation() == kTanH) {

  // candidate_hidden += bias_multiplier * bias
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_multiplier"), BnInOp2Blob("bias"),

  // candidate_hidden = tanh(update_gate)
  KernelUtil<device_type, T>::TanH(
      ctx.device_ctx, update_gate_blob->shape().elem_cnt(),
      candidate_hidden_blob->dptr<T>(), candidate_hidden_blob->mut_dptr<T>());

  // plus_op_out = hidden .* reset_gate
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(hidden->shape().At(0)), hidden_blob, reset_gate_blob,
      plus_op_out_blob);
  // 

  // rec_out = out
  BnInOp2Blob("rec_out")->CopyDataContentFrom(ctx.device_ctx, out_blob);
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_blob = BnInOp2Blob("out");
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* rec_out_diff_blob = BnInOp2Blob("rec_out_diff");
  // reuse memory
  Blob* plus_op_out_diff_blob = BnInOp2Blob("plus_op_out");

  if (this->op_conf().basic_rnn_conf().activation() == kTanH) {
    BasicGruKernelUtil<device_type, T>::ComputeTanHDiff(
        ctx.device_ctx, out_blob->shape().elem_cnt(), out_blob->dptr<T>(),
        out_diff_blob->dptr<T>(), rec_out_diff_blob->dptr<T>(),
        plus_op_out_diff_blob->mut_dptr<T>());
  } else if (this->op_conf().basic_rnn_conf().activation() == kSigmoid) {
    BasicGruKernelUtil<device_type, T>::ComputeSigmoidDiff(
        ctx.device_ctx, out_blob->shape().elem_cnt(), out_blob->dptr<T>(),
        out_diff_blob->dptr<T>(), rec_out_diff_blob->dptr<T>(),
        plus_op_out_diff_blob->mut_dptr<T>());
  } else {
    UNIMPLEMENTED();
  }

  // h2h_weight_diff = plus_op_out_diff * hidden
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       plus_op_out_diff_blob, hidden_blob,
                                       BnInOp2Blob("h2h_weight_diff"));

  // i2h_weight_diff = plus_op_out_diff * in
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       plus_op_out_diff_blob, BnInOp2Blob("in"),
                                       BnInOp2Blob("i2h_weight_diff"));

  // in_diff = plus_op_out_diff * i2h_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), plus_op_out_diff_blob, BnInOp2Blob("i2h_weight"),
      BnInOp2Blob("in_diff"));

  // bias_diff = bias_multiplier * plus_op_out_diff
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), BnInOp2Blob("bias_multiplier"), plus_op_out_diff_blob,
      BnInOp2Blob("bias_diff"));

  if (BnInOp2Blob("in")->col_id() != 0 || this->NeedExternalH0()
      || this->op_conf().basic_rnn_conf().is_init_hidden_trainable()) {
    // hidden_diff = plus_op_out_diff * h2h_weight
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(0), plus_op_out_diff_blob, BnInOp2Blob("h2h_weight"),
        this->GetHiddenDiffBlob(BnInOp2Blob));
  }
}
    DeviceCtx* ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* i2h_weight_blob = BnInOp2Blob("i2h_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, i2h_weight_blob, "i2h_weight",
      i2h_weight_blob->shape().At(0), i2h_weight_blob->shape().Count(1));
  Blob* h2h_weight_blob = BnInOp2Blob("h2h_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, h2h_weight_blob, "h2h_weight",
      h2h_weight_blob->shape().At(0), h2h_weight_blob->shape().Count(1));
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias"), "bias",
      BnInOp2Blob("bias")->shape().At(0), 1);
}

template<typename T>
class BasicGruKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void ComputeTanHDiff(DeviceCtx* ctx, int64_t n, const T* out,
                              const T* out_diff, const T* rec_out_diff,
                              T* plus_out_diff) {
    FOR_RANGE(int64_t, i, 0, n) {
      plus_out_diff[i] =
          (1 - out[i] * out[i]) * (out_diff[i] + rec_out_diff[i]);
    }
  }
  static void ComputeSigmoidDiff(DeviceCtx* ctx, int64_t n, const T* out,
                                 const T* out_diff, const T* rec_out_diff,
                                 T* plus_out_diff) {
    FOR_RANGE(int64_t, i, 0, n) {
      plus_out_diff[i] =
          out[i] * (1 - out[i]) * (out_diff[i] + rec_out_diff[i]);
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBasicGruConf, BasicGruKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
