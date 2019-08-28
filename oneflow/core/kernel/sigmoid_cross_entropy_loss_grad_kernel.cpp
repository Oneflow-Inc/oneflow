#include "oneflow/core/kernel/sigmoid_cross_entropy_loss_grad_kernel.h"


namespace oneflow {


template <DeviceType device_type, typename PredType, typename LabelType>
void SigmoidCrossEntropyLossGradKernel<device_type, PredType, LabelType>::ForwardDataContent(
       const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const SigmoidCrossEntropyLossGradOpConf& conf = 
          this->op_conf().sigmoid_cross_entropy_loss_grad_conf();

  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  Blob* pred_diff = BnInOp2Blob("loss_diff");
  
  const int64_t inner_dim_size = label->shape().Count(1);
  FOR_RANGE(int64_t, data_index, 0, prediction->shape().At(0)) {
    const int64_t offset =  data_index * inner_dim_size;
    const PredType* cur_pred = prediction->dptr<PredType>() + offset;
    const LabelType* cur_label = label->dptr<LabelType>() + offset;
    PredType* cur_pred_diff = pred_diff->dptr<PredType>() + offset;
    SigmoidCrossEntropyLossGradKernelUtil<device_type, PredType, LabelType>::Backward(
      ctx, conf, inner_dim_size, cur_pred, cur_label, cur_pred_diff);
  }
}



template<typename PredType, typename LabelType>
struct SigmoidCrossEntropyLossGradKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  
  static void Backward(DeviceCtx* ctx, const SigmoidCrossEntropyLossGradOpConf& conf,
                       const int64_t n, const PredType* prediction, const LabelType* label,
                       PredType* pred_diff) {
    FOR_RANGE(int64_t, index, 0, n) {
      if (label[index] != -1) {
        pred_diff[index] = 1.f / (1.f + expf(-prediction[index])) - label[index];
      }
    }
  }
};

// instantiate template declaration
template struct SigmoidCrossEntropyLossGradKernelUtil<DeviceType::kCPU, float, float>;

} // namespace oneflow
