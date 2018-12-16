#include "oneflow/core/kernel/center_loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void CenterLossKernel<device_type, PredType, LabelType>::VirtualLossForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction_blob = BnInOp2Blob("prediction");
  const Blob* label_blob = BnInOp2Blob("label");
  Blob* centers_blob = BnInOp2Blob("centers");
  Blob* piece_centers_blob = BnInOp2Blob("piece_centers");
  Blob* loss_blob = BnInOp2Blob("loss");
  Blob* prediction_diff_blob = BnInOp2Blob("prediction_diff");
  const int32_t n = prediction_blob->shape().At(0);
  const int32_t d = prediction_blob->shape().At(1);
  const auto& center_loss_op_conf = this->kernel_conf().op_attribute().op_conf().center_loss_conf();
  const int32_t num_classes = center_loss_op_conf.num_classes();
  const float alpha = center_loss_op_conf.alpha();

  CenterLossKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, prediction_blob->dptr<PredType>(), label_blob->dptr<LabelType>(), num_classes,
      d, n, alpha, piece_centers_blob->mut_dptr<PredType>(), centers_blob->mut_dptr<PredType>(),
      loss_blob->mut_dptr<PredType>(), prediction_diff_blob->mut_dptr<PredType>());
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
  static void Forward(DeviceCtx* ctx, const PredType* prediction, const LabelType* label,
                      const int32_t num_classes, const int32_t dim, const int32_t num_labels,
                      const float alpha, PredType* piece_centers, PredType* centers, PredType* loss,
                      PredType* prediction_diff) {
    // Lookup
    FOR_RANGE(int32_t, i, 0, num_labels) {
      CHECK(label[i] >= 0 && label[i] < num_classes);
      const PredType* from = centers + (label[i] * dim);
      PredType* to = piece_centers + (i * dim);
      std::copy(from, from + dim, to);
    }
    FOR_RANGE(int32_t, i, 0, num_labels) {
      loss[i] = 0;
      FOR_RANGE(int32_t, j, 0, dim) {
        // Forward
        int64_t index = i * dim + j;
        PredType diff = prediction[index] - piece_centers[index];
        loss[i] += 0.5 * diff * diff;
        // Update Centers
        PredType center_diff = (1 - alpha) * (piece_centers[index] - prediction[index]);
        centers[index] -= center_diff;
        // Backward
        prediction_diff[index] = prediction[index] - piece_centers[index];
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
                                kernel_conf.center_loss_conf().loss_conf().prediction_type(),
                                kernel_conf.center_loss_conf().loss_conf().label_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kCenterLossConf, CreateCenterLossKernel);

#define MAKE_ENTRY(data_type_pair, label_type_pair)                                        \
  template struct CenterLossKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                       OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
