#include "oneflow/core/kernel/center_loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void CenterLossKernel<device_type, PredType, LabelType>::VirtualLossForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // Forward
  const Blob* prediction_blob = BnInOp2Blob("prediction");
  const Blob* label_blob = BnInOp2Blob("label");
  const Blob* ones_multipiler_blob = BnInOp2Blob("ones_multipiler");
  Blob* centers_blob = BnInOp2Blob("centers");
  Blob* piece_centers_blob = BnInOp2Blob("piece_centers");
  Blob* forward_tmp_blob = BnInOp2Blob("forward_tmp");
  Blob* loss_blob = BnInOp2Blob("loss");
  const int32_t n = prediction_blob->shape().At(0);
  const int32_t d = prediction_blob->shape().At(1);
  const int32_t num_of_classes =
      this->kernel_conf().op_attribute().op_conf().center_loss_conf().num_of_classes();
  CenterLossKernelUtil<device_type, PredType, LabelType>::Lookup(
      ctx.device_ctx, prediction_blob->dptr<PredType>(), n, d, label_blob->dptr<LabelType>(), n,
      piece_centers_blob->mut_dptr<PredType>());
  KernelUtil<device_type, PredType>::Sub(ctx.device_ctx, n * d, prediction_blob->dptr<PredType>(),
                                         piece_centers_blob->dptr<PredType>(),
                                         forward_tmp_blob->mut_dptr<PredType>());
  KernelUtil<device_type, PredType>::Square(ctx.device_ctx, n * d,
                                            forward_tmp_blob->dptr<PredType>(),
                                            forward_tmp_blob->mut_dptr<PredType>());
  KernelUtil<device_type, PredType>::Scal(ctx.device_ctx, n * d, 0.5,
                                          forward_tmp_blob->mut_dptr<PredType>(), 1);
  KernelUtil<device_type, PredType>::OFGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, n, 1, d, 1.0, forward_tmp_blob->dptr<PredType>(),
      ones_multipiler_blob->dptr<PredType>(), 0.0, loss_blob->mut_dptr<PredType>());
  KernelUtil<device_type, PredType>::Scal(ctx.device_ctx, n, 1 / d, loss_blob->mut_dptr<PredType>(),
                                          1);

  // Update Centers
  const PredType alpha = this->kernel_conf().op_attribute().op_conf().center_loss_conf().alpha();
  KernelUtil<device_type, PredType>::Sub(ctx.device_ctx, n * d, prediction_blob->dptr<PredType>(),
                                         piece_centers_blob->dptr<PredType>(),
                                         forward_tmp_blob->mut_dptr<PredType>());
  KernelUtil<device_type, PredType>::Scal(ctx.device_ctx, n * d, (1 - alpha),
                                          forward_tmp_blob->mut_dptr<PredType>(), 1);
  CenterLossKernelUtil<device_type, PredType, LabelType>::SparseUpdate(
      ctx.device_ctx, forward_tmp_blob->dptr<PredType>(), n, d, label_blob->dptr<LabelType>(), n,
      centers_blob->mut_dptr<PredType>(), num_of_classes);

  // Backward
  Blob* prediction_diff_blob = BnInOp2Blob("prediction_diff");
  KernelUtil<device_type, PredType>::Sub(ctx.device_ctx, n * d, prediction_blob->dptr<PredType>(),
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
  KernelUtil<device_type, PredType>::InitializeWithProperConf(ctx, &centers_init_conf, 0,
                                                              BnInOp2Blob("centers"));
}

template<DeviceType device_type, typename PredType, typename LabelType>
void CenterLossKernel<device_type, PredType, LabelType>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* centers_blob = BnInOp2Blob("centers");
  KernelUtil<device_type, PredType>::InitializeWithDir(
      ctx, part_id, part_num, model_load_dir, centers_blob, "centers", centers_blob->shape().At(0),
      centers_blob->shape().Count(1));
}

template<typename PredType, typename LabelType>
struct CenterLossKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void Lookup(DeviceCtx* ctx, const PredType* in, const int32_t in_row_num,
                     const int32_t in_col_num, const LabelType* indices,
                     const int32_t num_of_indices, PredType* out) {
    FOR_RANGE(int32_t, i, 0, num_of_indices) {
      const int32_t idx = indices[i];
      CHECK(idx >= 0 && idx < in_row_num);
      const PredType* from = in + (idx * in_col_num);
      PredType* to = out + (i * in_col_num);
      std::copy(from, from + in_col_num, to);
    }
  }
  static void SparseUpdate(DeviceCtx* ctx, const PredType* diff, const int32_t diff_row_num,
                           const int32_t diff_col_num, const LabelType* indices,
                           int32_t num_of_indices, PredType* model, const int32_t model_row_num) {
    FOR_RANGE(int32_t, i, 0, num_of_indices) {
      FOR_RANGE(int32_t, j, 0, diff_col_num) {
        CHECK_LT(indices[i], model_row_num);
        model[indices[i] * diff_col_num + j] -= diff[i * diff_col_num + j];
      }
    }
  }
};

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf& CenterLossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.center_loss_conf().loss_conf();
}

namespace {

Kernel* CreateCenterLossKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define CENTER_LOSS_KERNEL_ENTRY(device_type, pred_type_pair, label_type_pair)                     \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair), OF_PP_PAIR_SECOND(label_type_pair)), \
   []() {                                                                                          \
     return new CenterLossKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair),                    \
                                 OF_PP_PAIR_FIRST(label_type_pair)>();                             \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(CENTER_LOSS_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                                kernel_conf.accuracy_conf().prediction_type(),
                                kernel_conf.accuracy_conf().label_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kCenterLossConf, CreateCenterLossKernel);

#define MAKE_ENTRY(data_type_pair, label_type_pair)                                        \
  template struct CenterLossKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                       OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow