#include "oneflow/core/kernel/basic_rnn_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& BasicRnnKernel<device_type, T>::GetRecurrentOpConf() const {
  return this->op_conf().basic_rnn_conf();
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* plus_op_out_blob = BnInOp2Blob("plus_op_out");
  Blob* out_blob = BnInOp2Blob("out");

  // plus_op_out = in * i2h_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), BnInOp2Blob("in"), BnInOp2Blob("i2h_weight"),
      plus_op_out_blob);

  // plus_op_out += hidden * h2h_weight
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       hidden_blob, BnInOp2Blob("h2h_weight"),
                                       plus_op_out_blob);

  // plus_op_out += bias_multiplier * bias
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_multiplier"), BnInOp2Blob("bias"),
      plus_op_out_blob);

  if (this->op_conf().basic_rnn_conf().activation() == kTanH) {
    KernelUtil<device_type, T>::TanH(
        ctx.device_ctx, out_blob->shape().elem_cnt(),
        plus_op_out_blob->dptr<T>(), out_blob->mut_dptr<T>());
  } else if (this->op_conf().basic_rnn_conf().activation() == kSigmoid) {
    KernelUtil<device_type, T>::Sigmoid(
        ctx.device_ctx, out_blob->shape().elem_cnt(),
        plus_op_out_blob->dptr<T>(), out_blob->mut_dptr<T>());
  } else {
    UNEXPECTED_RUN();
  }

  // rec_out = out
  BnInOp2Blob("rec_out")->CopyDataContentFrom<device_type>(ctx.device_ctx,
                                                           out_blob);
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDataIdFrom<device_type>(ctx.device_ctx,
                                                  BnInOp2Blob("in"));
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_blob = BnInOp2Blob("out");
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* rec_out_diff_blob = BnInOp2Blob("rec_out_diff");
  // reuse memory
  Blob* plus_op_out_diff_blob = BnInOp2Blob("plus_op_out");

  if (this->op_conf().basic_rnn_conf().activation() == kTanH) {
    BasicRnnKernelUtil<device_type, T>::ComputeTanHDiff(
        ctx.device_ctx, out_blob->shape().elem_cnt(), out_blob->dptr<T>(),
        out_diff_blob->dptr<T>(), rec_out_diff_blob->dptr<T>(),
        plus_op_out_diff_blob->mut_dptr<T>());
  } else if (this->op_conf().basic_rnn_conf().activation() == kSigmoid) {
    BasicRnnKernelUtil<device_type, T>::ComputeSigmoidDiff(
        ctx.device_ctx, out_blob->shape().elem_cnt(), out_blob->dptr<T>(),
        out_diff_blob->dptr<T>(), rec_out_diff_blob->dptr<T>(),
        plus_op_out_diff_blob->mut_dptr<T>());
  } else {
    UNEXPECTED_RUN();
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

  // hidden_diff = plus_op_out_diff * h2h_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), plus_op_out_diff_blob, BnInOp2Blob("h2h_weight"),
      this->GetHiddenDiffBlob(BnInOp2Blob));
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::VirtualInitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_rnn_conf(),
                        i2h_weight_initializer),
      random_seed_gen(), BnInOp2Blob("i2h_weight"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_rnn_conf(),
                        h2h_weight_initializer),
      random_seed_gen(), BnInOp2Blob("h2h_weight"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_rnn_conf(), bias_initializer),
      random_seed_gen(), BnInOp2Blob("bias"));
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::VirtualInitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* i2h_weight_blob = BnInOp2Blob("i2h_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir, i2h_weight_blob,
      "i2h_weight", i2h_weight_blob->shape().At(0),
      i2h_weight_blob->shape().Count(1));
  Blob* h2h_weight_blob = BnInOp2Blob("h2h_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir, h2h_weight_blob,
      "h2h_weight", h2h_weight_blob->shape().At(0),
      h2h_weight_blob->shape().Count(1));
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias"),
      "bias", BnInOp2Blob("bias")->shape().At(0), 1);
}

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::InitModelTmpBlobs(
    const KernelCtx& ctx, const ParallelContext* parallel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf bias_multiplier_fill_conf;
  bias_multiplier_fill_conf.mutable_constant_conf()->set_value(1.f);
  KernelUtil<device_type, T>::Initialize(ctx.device_ctx,
                                         bias_multiplier_fill_conf, 0,
                                         BnInOp2Blob("bias_multiplier"));
}

template<typename T>
class BasicRnnKernelUtil<DeviceType::kCPU, T> final {
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

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBasicRnnConf, BasicRnnKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
