#include "oneflow/core/kernel/smooth_l1_loss_kernel.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void SmoothL1LossKernel<device_type, PredType, LabelType>::VirtualLossForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  const Blob* inside_weights = BnInOp2Blob("inside_weights");
  const Blob* outside_weights = BnInOp2Blob("outside_weights");
  Blob* const_all_one = BnInOp2Blob("const_all_one");
  Blob* loss_buf = BnInOp2Blob("loss_buf");
  Blob* loss = BnInOp2Blob("loss");
  Blob* pred_diff_blob = BnInOp2Blob(GenDiffBn("prediction"));
  auto kernel_conf = this->kernel_conf();
  const float beta = kernel_conf.op_attribute().op_conf().smooth_l1_loss_conf().beta();
  const float scale = kernel_conf.op_attribute().op_conf().smooth_l1_loss_conf().scale();
  int64_t N = BnInOp2Blob("prediction")->shape().At(0);
  int64_t D = BnInOp2Blob("prediction")->shape().Count(1);

  if (const_all_one != nullptr) {
    Memset<device_type>(ctx.device_ctx, const_all_one->mut_dptr<PredType>(), 1,
                        const_all_one->TotalByteSize());
  }

  SmoothL1LossKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, N, D, prediction->dptr<PredType>(), label->dptr<LabelType>(),
      inside_weights->dptr<int8_t>(), outside_weights->dptr<int8_t>(),
      const_all_one->dptr<PredType>(), beta, scale, loss_buf->mut_dptr<PredType>(),
      loss->mut_dptr<PredType>());

  SmoothL1LossKernelUtil<device_type, PredType, LabelType>::Backward(
      ctx.device_ctx, N, D, prediction->dptr<PredType>(), label->dptr<LabelType>(),
      inside_weights->dptr<int8_t>(), outside_weights->dptr<int8_t>(), beta, scale,
      pred_diff_blob->mut_dptr<PredType>());
}

template<typename PredType, typename LabelType>
struct SmoothL1LossKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int64_t N, const int64_t D, const PredType* prediction,
                      const LabelType* label, const int8_t* inside_weights,
                      const int8_t* outside_weights, const PredType* const_all_one,
                      const float beta, const float scale, PredType* loss_buf, PredType* loss) {
    int64_t elem_cnt = N * D;
    for (int i = 0; i < elem_cnt; i++) {
      PredType x = inside_weights[i] * (prediction[i] - label[i]);
      PredType abs_x = abs(x);
      if (abs_x < beta) {
        loss_buf[i] = 0.5 * x * x / beta;
      } else {
        loss_buf[i] = abs_x - 0.5 * beta;
      }
      loss_buf[i] *= scale / elem_cnt * outside_weights[i];
    }
    KernelUtil<DeviceType::kCPU, PredType>::Dot(ctx, N * D, loss_buf, 1, const_all_one, 1, loss);
  }
  static void Backward(DeviceCtx* ctx, const int64_t N, const int64_t D, const PredType* prediction,
                       const LabelType* label, const int8_t* inside_weights,
                       const int8_t* outside_weights, const float beta, const float scale,
                       PredType* in_diff) {
    int64_t elem_cnt = N * D;
    for (int i = 0; i < elem_cnt; i++) {
      PredType x = inside_weights[i] * (prediction[i] - label[i]);
      PredType abs_x = abs(x);
      if (abs_x < beta) {
        in_diff[i] = x / beta;
      } else {
        in_diff[i] = x > 0 ? 1 : -1;
      }
      in_diff[i] *= scale / elem_cnt * outside_weights[i];
    }
  }
};

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf& SmoothL1LossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.smooth_l1_loss_conf().loss_conf();
}

namespace {

Kernel* CreateSmoothL1LossKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define SMOOTH_L1_LOSS_KERNEL_ENTRY(device_type, pred_type_pair, label_type_pair)                  \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair), OF_PP_PAIR_SECOND(label_type_pair)), \
   []() {                                                                                          \
     return new SmoothL1LossKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair),                  \
                                   OF_PP_PAIR_FIRST(label_type_pair)>();                           \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SMOOTH_L1_LOSS_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                                kernel_conf.smooth_l1_loss_conf().loss_conf().prediction_type(),
                                kernel_conf.smooth_l1_loss_conf().loss_conf().label_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kSmoothL1LossConf, CreateSmoothL1LossKernel);

}  // namespace oneflow
