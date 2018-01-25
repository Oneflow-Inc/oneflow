#include "oneflow/core/kernel/basic_rnn_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* ht_1_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* plus_op_out_blob = BnInOp2Blob("plus_op_out");
  Blob* ht_blob = BnInOp2Blob("ht");

  // plus_op_out = in * i2h_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), BnInOp2Blob("in"), BnInOp2Blob("i2h_weight"),
      plus_op_out_blob);

  // plus_op_out += ht_1 * h2h_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(1), ht_1_blob, BnInOp2Blob("h2h_weight"),
      plus_op_out_blob);

  // plus_op_out += bias_multiplier * bias
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_multiplier"), BnInOp2Blob("bias"),
      plus_op_out_blob);

  if (this->op_conf().recurrent_conf().activation() == kTanH) {
    BasicRnnKernelUtil<device_type, T>::TanH(
        ctx.device_ctx, ht_blob->shape().elem_cnt(),
        plus_op_out_blob->dptr<T>(), ht_blob->mut_dptr<T>());
  } else if (this->op_conf().recurrent_conf().activation() == kSigmoid) {
    BasicRnnKernelUtil<device_type, T>::Sigmoid(
        ctx.device_ctx, ht_blob->shape().elem_cnt(),
        plus_op_out_blob->dptr<T>(), ht_blob->mut_dptr<T>());
  } else {
    UNEXPECTED_RUN();
  }

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
  const Blob* ht_blob = BnInOp2Blob("ht");
  const Blob* ht_1_blob = this->GetHiddenBlob(BnInOp2Blob);
  const Blob* ht_diff_blob = BnInOp2Blob("ht_diff");
  const Blob* rec_ht_diff_blob = BnInOp2Blob("rec_ht_diff");
  // reuse memory
  Blob* plus_op_out_diff_blob = BnInOp2Blob("plus_op_out");

  if (this->op_conf().recurrent_conf().activation() == kTanH) {
    BasicRnnKernelUtil<device_type, T>::ComputeTanHDiff(
        ctx.device_ctx, ht_blob->shape().elem_cnt(), ht_blob->dptr<T>(),
        ht_diff_blob->dptr<T>(), rec_ht_diff_blob->dptr<T>(),
        plus_op_out_diff_blob->mut_dptr<T>());
  } else if (this->op_conf().recurrent_conf().activation() == kSigmoid) {
    BasicRnnKernelUtil<device_type, T>::ComputeSigmoidDiff(
        ctx.device_ctx, ht_blob->shape().elem_cnt(), ht_blob->dptr<T>(),
        ht_diff_blob->dptr<T>(), rec_ht_diff_blob->dptr<T>(),
        plus_op_out_diff_blob->mut_dptr<T>());
  } else {
    UNEXPECTED_RUN();
  }

  // h2h_weight_diff = plus_op_out_diff * ht_1
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       plus_op_out_diff_blob, ht_1_blob,
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

  if (this->Ish0Model() && BnInOp2Blob("rec_ht_diff")->col_id() == 0) {
    // h0_diff = plus_op_out_diff * h2h_weight
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(0),
        static_cast<T>(0), plus_op_out_diff_blob, BnInOp2Blob("h2h_weight"),
        BnInOp2Blob("h0_diff"));
  }
}

#define INITIALZE_BLOB(initializer, base_initializer, bn)                      \
  const InitializerConf* bn##initializer_conf = OF_PB_POINTER_GET(             \
      this->op_conf().recurrent_conf().basic_rnn_cell(), initializer);         \
  if (bn##initializer_conf == nullptr) {                                       \
    bn##initializer_conf =                                                     \
        OF_PB_POINTER_GET(this->op_conf().recurrent_conf(), base_initializer); \
  }                                                                            \
  KernelUtil<device_type, T>::InitializeWithProperConf(                        \
      ctx.device_ctx, bn##initializer_conf, random_seed_gen(),                 \
      BnInOp2Blob(#bn));

template<DeviceType device_type, typename T>
void BasicRnnKernel<device_type, T>::VirtualInitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  INITIALZE_BLOB(i2h_weight_initializer, weight_initializer, i2h_weight);
  INITIALZE_BLOB(h2h_weight_initializer, weight_initializer, h2h_weight);
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(this->op_conf().recurrent_conf(), bias_initializer),
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
  static void Add(DeviceCtx* ctx, int64_t n, const T* x, const T* y, T* z) {
    FOR_RANGE(int64_t, i, 0, n) { z[i] = x[i] + y[i]; }
  }
  static void Sigmoid(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
    FOR_RANGE(int64_t, i, 0, n) {
      y[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x[i]));
    }
  }
  static void TanH(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
    T one = static_cast<T>(1);
    T two = static_cast<T>(2);
    FOR_RANGE(int64_t, i, 0, n) {
      y[i] = two / (one + std::exp(-two * x[i])) - one;
    }
  }
  static void ComputeTanHDiff(DeviceCtx* ctx, int64_t n, const T* ht,
                              const T* ht_diff, const T* rec_ht_diff,
                              T* plus_out_diff) {
    FOR_RANGE(int64_t, i, 0, n) {
      plus_out_diff[i] = (1 - ht[i] * ht[i]) * (ht_diff[i] + rec_ht_diff[i]);
    }
  }
  static void ComputeSigmoidDiff(DeviceCtx* ctx, int64_t n, const T* ht,
                                 const T* ht_diff, const T* rec_ht_diff,
                                 T* plus_out_diff) {
    FOR_RANGE(int64_t, i, 0, n) {
      plus_out_diff[i] = ht[i] * (1 - ht[i]) * (ht_diff[i] + rec_ht_diff[i]);
    }
  }
};

template class BasicRnnKernelUtil<DeviceType::kCPU, float>;
template class BasicRnnKernelUtil<DeviceType::kCPU, double>;

#define INSTANTIATE_KERNEL(device_type, data_type_pair) \
  template class BasicRnnKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
