#include "oneflow/core/kernel/center_loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void CenterLossKernel<device_type, PredType, LabelType>::VirtualLossForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // Forward
  const Blob* prediction_blob = BnInOp2Blob("prediction");
  const Blob* label_blob = BnInOp2Blob("label");
  const Blob* centers_blob = BnInOp2Blob("centers");
  const Blob* ones_multipiler_blob = BnInOp2Blob("ones_multipiler");
  Blob* piece_centers_blob = BnInOp2Blob("piece_centers");
  Blob* forward_tmp_blob = BnInOp2Blob("forward_tmp");
  Blob* loss_blob = BnInOp2Blob("loss");
  int32_t n = prediction_blob->shape().At(0);
  int32_t d = prediction_blob->shape().At(1);
  CenterLossKernelUtil<device_type, PredType, LabelType>::Gather(
      centers_blob->dptr<PredType>(), label_blob->dptr<LabelType>(),
      piece_centers_blob->mut_dptr<PredType>());
  KernelUtil<device_type, PredType>::Sub(ctx, n * d, prediction_blob->dptr<PredType>(),
                                         piece_centers_blob->dptr<PredType>(),
                                         forward_tmp_blob->mut_dptr<PredType>());
  KernelUtil<device_type, PredType>::Square(ctx, n * d, forward_tmp_blob->dptr<PredType>(),
                                            forward_tmp_blob->mut_dptr<PredType>());
  KernelUtil<device_type, PredType>::Scal(ctx, n * d, 0.5, forward_tmp_blob->mut_dptr<PredType>(),
                                          1);
  KernelUtil<device_type, PredType>::OFGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, n, 1, d, 1.0, forward_tmp_blob->dptr<PredType>(),
      ones_multipiler_blob->dptr<PredType>(), 0.0, loss_blob->mut_dptr<PredType>());
  KernelUtil<device_type, PredType>::Scal(ctx, n, 1 / d, loss_blob->mut_dptr<PredType>(), 1);

  // update forward model
  const PredType alpha = this->kernel_conf().op_attribute().op_conf().center_loss_conf().alpha();
  KernelUtil<device_type, PredType>::Sub(ctx, n * d, prediction_blob->dptr<PredType>(),
                                         piece_centers_blob->dptr<PredType>(),
                                         forward_tmp_blob->mut_dptr<PredType>());
  KernelUtil<device_type, PredType>::Scal(ctx, n * d, (1 - alpha),
                                          forward_tmp_blob->mut_dptr<PredType>(), 1);
  CenterLossKernelUtil<device_type, PredType, LabelType>::SparseUpdate(
      label_blob->dptr<LabelType>(), forward_tmp_blob->dptr<PredType>(),
      centers_blob->mut_dptr<PredType>());

  // Backward
  Blob* prediction_diff_blob = BnInOp2Blob("prediction_diff");
  KernelUtil<device_type, PredType>::Sub(ctx, n * d, prediction_blob->dptr<PredType>(),
                                         piece_centers_blob->dptr<PredType>(),
                                         prediction_diff_blob->mut_dptr<PredType>());
}

template<DeviceType device_type, typename PredType, typename LabelType>
void CenterLossKernel<device_type, PredType, LabelType>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf ones_multipiler_initializer_conf;
  ones_multipiler_initializer_conf.mutable_constant_conf()->set_value(1.0f);
  KernelUtil<device_type, PredType>::InitializeWithConf(ctx, ones_multipiler_initializer_conf, 0,
                                                        BnInOp2Blob("ones_multiplier"));
}

template<DeviceType device_type, typename PredType, typename LabelType>
void CenterLossKernel<device_type, PredType, LabelType>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf centers_init_conf;
  centers_init_conf.mutable_constant_conf()->set_value(0.0);
  KernelUtil<device_type, PredType>::InitializaWithProperConf(ctx, &centers_init_conf, 0,
                                                              BnInOp2Blob("centers"));
}

template<DeviceType device_type, typename PredType, typename LabelType>
void CenterLossKernel<device_type, PredType, LabelType>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // centers are always initialized to zero
  InitializerConf centers_init_conf;
  centers_init_conf.mutable_constant_conf()->set_value(0.0f);
  KernelUtil<device_type, PredType>::InitializaWithProperConf(ctx, &centers_init_conf, 0,
                                                              BnInOp2Blob("centers"));
}

template<typename PredType, typename LabelType>
struct CenterLossKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void Gather(const PredType* centers_ptr, const LabelType* label_ptr,
                     PredType* piece_centers_ptr) {
    // TODO
  }
  static void SparseUpdate(int32_t n, const LabelType* label_ptr, PredType* center_diff_ptr,
                           PredType* centers_ptr) {
    // TODO
  }
};

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf& CenterLossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.center_loss_conf().loss_conf();
}

}  // namespace oneflow