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
  Blob* reset_gate_data_blob = BnInOp2Blob("reset_gate_data");
  Blob* update_gate_data_blob = BnInOp2Blob("update_gate_data");
  Blob* candidate_hidden_data_blob = BnInOp2Blob("candidate_hidden_data");
  Blob* candidate_hidden_out_blob = BnInOp2Blob("candidate_hidden_out");
  Blob* reset_gate_out_blob = BnInOp2Blob("reset_gate_out");
  Blob* update_gate_out_blob = BnInOp2Blob("update_gate_out");
  Blob* update_mul_hidden_blob = BnInOp2Blob("update_mul_hidden");

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
      static_cast<T>(1), BnInOp2Blob("bias_multiplier_r"),
      BnInOp2Blob("bias_r"), reset_gate_data_blob);
  // reset_gate_out = sigmoid(reset_gate_data)
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, reset_gate_data_blob->shape().elem_cnt(),
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
      static_cast<T>(1), BnInOp2Blob("bias_multiplier_z"),
      BnInOp2Blob("bias_z"), update_gate_data_blob);
  // update_gate_out = sigmoid(update_gate_data)
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, update_gate_data_blob->shape().elem_cnt(),
      update_gate_data_blob->dptr<T>(), update_gate_out_blob->mut_dptr<T>());

  // candidate_hidden_data = hidden .*reset_gate_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(reset_gate_out_blob->shape().elem_cnt()),
      reset_gate_out_blob->dptr<T>(), hidden_blob->dptr<T>(),
      candidate_hidden_data_blob->mut_dptr<T>());

  // candidate_hidden_data *= h2h_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), candidate_hidden_data_blob, BnInOp2Blob("h2h_weght"),
      candidate_hidden_data_blob);

  // candidate_hidden_data += in * i2h_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("in"), BnInOp2Blob("i2h_weight"),
      candidate_hidden_data_blob);

  // candidate_hidden_data += bias_multiplier * bias
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_multiplier"), BnInOp2Blob("bias"),
      candidate_hidden_data_blob);

  // candidate_hidden_out = tanh(candidate_hidden_data)
  KernelUtil<device_type, T>::TanH(
      ctx.device_ctx, candidate_hidden_data_blob->shape().elem_cnt(),
      candidate_hidden_data_blob->dptr<T>(),
      candidate_hidden_out_blob->mut_dptr<T>());

  // plus_op_out = candidate_hidden_out .* update_gate_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(update_gate_out_blob->shape().elem_cnt()),
      candidate_hidden_out_blob->dptr<T>(), update_gate_out_blob->dptr<T>(),
      plus_op_out_blob->mut_dptr<T>());
  // update_mul_hidden = hidden .* update_gate_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(update_gate_out_blob->shape().elem_cnt()),
      hidden_blob->dptr<T>(), update_gate_out_blob->dptr<T>(),
      update_mul_hidden_blob->mut_dptr<T>());

  // plus_op_out -= update_mul_hidden
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx,
      static_cast<T>(update_mul_hidden_blob->shape().elem_cnt()),
      static_cast<T>(-1), update_mul_hidden_blob->dptr<T>(), static_cast<T>(1),
      plus_op_out_blob->mut_dptr<T>(), static_cast<T>(1));
  // plus_op_out += hidden
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, static_cast<T>(hidden_blob->shape().elem_cnt()),
      static_cast<T>(1), hidden_blob->dptr<T>(), static_cast<T>(1),
      plus_op_out_blob->mut_dptr<T>(), static_cast<T>(1));
  // out = plus_op_out
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, plus_op_out_blob);
  // rec_out = plus_op_out
  BnInOp2Blob("rec_out")->CopyDataContentFrom(ctx.device_ctx, plus_op_out_blob);
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* in_diff_blob = BnInOp2Blob("in_diff");
  const Blob* rec_out_diff_blob = BnInOp2Blob("rec_out_diff");
  const Blob* update_gate_out_blob = BnInOp2Blob("update_gate_out");
  const Blob* update_gate_out_diff_blob = BnInOp2Blob("update_gate_out_diff");
  const Blob* candidate_hidden_data_blob = BnInOp2Blob("candidate_hidden_data");
  const Blob* candidate_hidden_out_blob = BnInOp2Blob("candidate_hidden_out");
  Blob* candidate_hidden_data_diff_blob =
      BnInOp2Blob("candidate_hidden_data_diff");

  Blob* candidate_hidden_out_diff_blob =
      BnInOp2Blob("candidate_hidden_out_diff");

  Blob* plus_op_out_diff_blob = BnInOp2Blob("plus_op_out_diff");

  BasicGruKernelUtil<device_type, T>::ComputeSigmoidDiff(
      ctx.device_ctx, out_blob->shape().elem_cnt(), out_blob->dptr<T>(),
      out_diff_blob->dptr<T>(), rec_out_diff_blob->dptr<T>(),
      plus_op_out_diff_blob->mut_dptr<T>());

  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, update_gate_out_blob->shape().elem_cnt(),
      update_gate_out_blob->dptr<T>(), plus_op_out_diff_blob->dptr<T>(),
      candidate_hidden_out_diff_blob->mut_dptr<T>());

  /*BasicGruKernelUtil<device_type, T>::ComputeTanHDiff(
      ctx.device_ctx, candidate_hidden_out_blob->shape().elem_cnt(),
      candidate_hidden_out_blob->dptr<T>(),
      candidate_hidden_out_diff_blob->dptr<T>(),
      candidate_hidden_data_diff_blob->mut_dptr<T>());

  BasicGruKernelUtil<device_type, T>::ComputeMulDiff(
      ctx.device_ctx, update_gate_out_blob->shape().elem_cnt(),
      candidate_hidden_out_blob->dptr<T>(), plus_op_out_diff_blob->dptr<T>(),
      update_gate_out_diff_blob->mut_dptr<T>());*/
}

template<typename T>
struct BasicGruKernelUtil<DeviceType::kCPU, T> {
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
