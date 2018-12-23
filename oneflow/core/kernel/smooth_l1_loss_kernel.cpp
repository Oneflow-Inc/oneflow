#include "oneflow/core/kernel/smooth_l1_loss_kernel.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SmoothL1LossKernel<device_type, T>::VirtualLossForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  const Blob* inside_weights = BnInOp2Blob("inside_weights");
  const Blob* outside_weights = BnInOp2Blob("outside_weights");
  Blob* loss = BnInOp2Blob("loss");
  Blob* pred_diff_blob = BnInOp2Blob(GenDiffBn("prediction"));
  auto kernel_conf = this->kernel_conf();
  const float beta = kernel_conf.op_attribute().op_conf().smooth_l1_loss_conf().beta();
  const float scale = kernel_conf.op_attribute().op_conf().smooth_l1_loss_conf().scale();
  int32_t elem_cnt = BnInOp2Blob("prediction")->shape().elem_cnt();

  Memset<device_type>(ctx.device_ctx, loss->mut_dptr(), 0, loss->ByteSizeOfDataContentField());
  Memset<device_type>(ctx.device_ctx, pred_diff_blob->mut_dptr(), 0,
                      pred_diff_blob->ByteSizeOfDataContentField());

  SmoothL1LossKernelUtil<device_type, T>::Forward(
      ctx.device_ctx, elem_cnt, prediction->dptr<T>(), label->dptr<T>(), inside_weights->dptr<T>(),
      outside_weights->dptr<T>(), beta, scale, loss->mut_dptr<T>());

  SmoothL1LossKernelUtil<device_type, T>::Backward(
      ctx.device_ctx, elem_cnt, prediction->dptr<T>(), label->dptr<T>(), inside_weights->dptr<T>(),
      outside_weights->dptr<T>(), beta, scale, pred_diff_blob->mut_dptr<T>());
}

template<typename T>
struct SmoothL1LossKernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int32_t elem_cnt, const T* prediction, const T* label,
                      const T* inside_weights, const T* outside_weights, const float beta,
                      const float scale, T* loss) {
    for (int i = 0; i < elem_cnt; i++) {
      T x = inside_weights[i] * (prediction[i] - label[i]);
      T abs_x = std::abs(x);
      if (abs_x < beta) {
        loss[i] = 0.5 * x * x / beta;
      } else {
        loss[i] = abs_x - 0.5 * beta;
      }
      loss[i] *= scale * outside_weights[i];
    }
  }
  static void Backward(DeviceCtx* ctx, const int32_t elem_cnt, const T* prediction, const T* label,
                       const T* inside_weights, const T* outside_weights, const float beta,
                       const float scale, T* in_diff) {
    for (int i = 0; i < elem_cnt; i++) {
      T x = inside_weights[i] * (prediction[i] - label[i]);
      T abs_x = std::abs(x);
      if (abs_x < beta) {
        in_diff[i] = x / beta;
      } else {
        in_diff[i] = (x > ZeroVal<T>::value) - (x < ZeroVal<T>::value);
      }
      in_diff[i] *= scale * inside_weights[i] * outside_weights[i];
    }
  }
};

template<DeviceType device_type, typename T>
const LossKernelConf& SmoothL1LossKernel<device_type, T>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.smooth_l1_loss_conf().loss_conf();
}

namespace {

Kernel* CreateSmoothL1LossKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define SMOOTH_L1_LOSS_KERNEL_ENTRY(device_type, pred_type_pair) \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(pred_type_pair)),   \
   []() { return new SmoothL1LossKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair)>(); }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SMOOTH_L1_LOSS_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                                kernel_conf.smooth_l1_loss_conf().loss_conf().prediction_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kSmoothL1LossConf, CreateSmoothL1LossKernel);

}  // namespace oneflow
