#include "oneflow/core/kernel/fully_connected_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
void BlasMatrixMatrix(const KernelCtx& ctx, const enum CBLAS_TRANSPOSE trans_a,
                      const enum CBLAS_TRANSPOSE trans_b, const T alpha,
                      const T beta, const Blob* a, const Blob* b, Blob* c) {
  const int m = c->shape().At(0);
  const int n = c->shape().Count(1);
  const int k =
      (trans_a == CblasNoTrans) ? a->shape().Count(1) : a->shape().At(0);

  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;

  KernelUtil<device_type, T>::Gemm(
      ctx.device_ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha,
      a->dptr<T>(), lda, b->dptr<T>(), ldb, beta, c->mut_dptr<T>(), ldc);
}

}  // namespace

template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* inputs_blob = BnInOp2Blob("inputs");
  const Blob* weights_blob = BnInOp2Blob("weights");
  Blob* outputs_blob = BnInOp2Blob("outputs");

  // outputs = inputs * weights
  BlasMatrixMatrix<device_type, T>(ctx, CblasNoTrans, CblasTrans,
                                   static_cast<T>(1.0), static_cast<T>(0.0),
                                   inputs_blob, weights_blob, outputs_blob);

  const Blob* biases_blob = BnInOp2Blob("biases");
  const Blob* biases_mul_blob = BnInOp2Blob("biases_multiplier");

  // outputs = biases_multiplier * biases + outputs
  BlasMatrixMatrix<device_type, T>(ctx, CblasNoTrans, CblasNoTrans,
                                   static_cast<T>(1.0), static_cast<T>(1.0),
                                   biases_mul_blob, biases_blob, outputs_blob);
}

template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* inputs_blob = BnInOp2Blob("inputs");
  const Blob* outputs_diff_blob = BnInOp2Blob("outputs_diff");
  Blob* inputs_diff_blob = BnInOp2Blob("inputs_diff");

  const Blob* weights_blob = BnInOp2Blob("weights");
  Blob* weight_diff_blob = BnInOp2Blob("weights_diff");

  // weights_diff = outputs_diff * inputs
  BlasMatrixMatrix<device_type, T>(
      ctx, CblasTrans, CblasNoTrans, static_cast<T>(1.0), static_cast<T>(0.0),
      outputs_diff_blob, inputs_blob, weight_diff_blob);

  // inputs_diff = outputs_diff * weights
  if (inputs_diff_blob != nullptr) {
    BlasMatrixMatrix<device_type, T>(
        ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1.0),
        static_cast<T>(0.0), outputs_diff_blob, weights_blob, inputs_diff_blob);
  }

  const Blob* biases_mul_blob = BnInOp2Blob("biases_multiplier");
  Blob* biases_diff_blob = BnInOp2Blob("biases_diff");

  // biases_diff = biases_multiplier * outputs_diff
  BlasMatrixMatrix<device_type, T>(
      ctx, CblasTrans, CblasNoTrans, static_cast<T>(1.0), static_cast<T>(0.0),
      biases_mul_blob, outputs_diff_blob, biases_diff_blob);
}

template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(this->op_conf().fully_connected_conf(),
                        weights_initializer),
      random_seed_gen(), BnInOp2Blob("weights"));

  if (this->op_conf().fully_connected_conf().has_biases_initializer()) {
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx.device_ctx,
        &(this->op_conf().fully_connected_conf().biases_initializer()),
        random_seed_gen(), BnInOp2Blob("biases"));
  } else {
    InitializerConf biases_initializer_conf;
    biases_initializer_conf.mutable_constant_conf()->set_value(0.0f);
    KernelUtil<device_type, T>::Initialize(
        ctx.device_ctx, biases_initializer_conf, 0, BnInOp2Blob("biases"));
  }
}
template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::InitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weights_blob = BnInOp2Blob("weightes");
  int32_t dim_num = this->op_conf().fully_connected_conf().num_outputs();
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir, weights_blob,
      "weights", dim_num, weights_blob->shape().Count(1));
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob("biases"),
      "biases", dim_num, 1);
}

template<DeviceType device_type, typename T>
void FullyConnectedKernel<device_type, T>::InitModelTmpBlobs(
    const KernelCtx& ctx, const ParallelContext* parallel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf biases_multiplier_initializer_conf;
  if (this->op_conf().fully_connected_conf().has_biases_initializer()) {
    biases_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
  } else {
    biases_multiplier_initializer_conf.mutable_constant_conf()->set_value(0.0f);
  }
  KernelUtil<device_type, T>::Initialize(ctx.device_ctx,
                                         biases_multiplier_initializer_conf, 0,
                                         BnInOp2Blob("biases_multiplier"));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kFullyConnectedConf,
                           FullyConnectedKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
