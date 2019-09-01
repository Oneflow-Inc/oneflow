#include "oneflow/core/kernel/sigmoid_cross_entropy_loss_kernel.h"
#include "oneflow/core/kernel/sparse_cross_entropy_loss_kernel.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void SigmoidCrossEntropyLossKernel<device_type, PredType, LabelType>::VirtualLossForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const SigmoidCrossEntropyLossOpConf& conf = this->op_conf().sigmoid_cross_entropy_loss_conf();
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  Blob* loss_buf = BnInOp2Blob("loss_buf");
  Blob* tmp_storage = BnInOp2Blob("sum_buf");
  const size_t tmp_storage_byte_size = static_cast<size_t>(tmp_storage->shape().elem_cnt());
  Blob* count = BnInOp2Blob("count");
  Blob* label_num = BnInOp2Blob("label_num");
  Blob* loss = BnInOp2Blob("loss");
  int64_t data_dim = label->shape().Count(1);
  int64_t data_offset = 0;
  FOR_RANGE(int64_t, data_index, 0, prediction->shape().At(0)) {
    data_offset = data_dim * data_index;
    const PredType* prediction_offset = prediction->dptr<PredType>() + data_offset;
    const LabelType* label_offset = label->dptr<LabelType>() + data_offset;
    SigmoidCrossEntropyLossKernelUtil<device_type, PredType, LabelType>::Forward(
        ctx.device_ctx, conf, data_dim, prediction_offset, label_offset,
        loss_buf->mut_dptr<PredType>(), tmp_storage->mut_dptr<PredType>(), tmp_storage_byte_size,
        count->mut_dptr<PredType>(), label_num->mut_dptr<PredType>(),
        loss->mut_dptr<PredType>() + data_index);
  }
}

template<DeviceType device_type, typename PredType, typename LabelType>
const LossKernelConf&
SigmoidCrossEntropyLossKernel<device_type, PredType, LabelType>::GetLossKernelConf(
    const KernelConf& kernel_conf) const {
  return kernel_conf.sigmoid_cross_entropy_loss_conf().loss_conf();
}

template<typename PredType, typename LabelType>
struct SigmoidCrossEntropyLossKernelUtil<DeviceType::kCPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const SigmoidCrossEntropyLossOpConf& conf, const int64_t n,
                      const PredType* prediction, const LabelType* label, PredType* loss_buf,
                      PredType* tmp_storage, const size_t tmp_storage_byte_size, PredType* count,
                      PredType* label_num, PredType* loss) {
    loss_buf[0] = 0;
    loss[0] = 0;
    label_num[0] = 0;
    FOR_RANGE(int64_t, index, 0, n) {
      if (label[index] != -1) {
        loss_buf[0] +=
            -1 * prediction[index] * (label[index] - (prediction[index] >= 0))
            + logf(1 + expf(prediction[index] - 2 * prediction[index] * (prediction[index] >= 0)));
        label_num[0] += 1;
      }
    }
    loss_buf[0] *= static_cast<PredType>(conf.scale());
    if (conf.normalize()) {
      if (label_num[0] == 0) { label_num[0] = 1e-5; }
      loss[0] = loss_buf[0] / label_num[0];
    }
  }

  static void Backward(DeviceCtx* ctx, const SigmoidCrossEntropyLossOpConf& conf, const int64_t n,
                       const PredType* prediction, const LabelType* label,
                       const PredType* label_num, PredType* pred_diff) {
    FOR_RANGE(int64_t, index, 0, n) {
      if (label[index] != -1) {
        pred_diff[index] = 1.f / (1.f + expf(-prediction[index])) - label[index];
        pred_diff[index] *= static_cast<PredType>(conf.scale());
        if (conf.normalize()) { pred_diff[index] /= label_num[0]; }
      }
    }
  }
};

#define REGISTER_SIGMOID_CROSS_ENTROPY_LOSS_KERNEL(pred_type, label_type)                  \
  REGISTER_KERNEL_WITH_PRED_AND_LABEL(                                                     \
      OperatorConf::kSigmoidCrossEntropyLossConf, DeviceType::kGPU, pred_type, label_type, \
      SigmoidCrossEntropyLossKernel<DeviceType::kGPU, pred_type, label_type>)

REGISTER_SIGMOID_CROSS_ENTROPY_LOSS_KERNEL(float, int32_t);
REGISTER_SIGMOID_CROSS_ENTROPY_LOSS_KERNEL(float, int64_t);
REGISTER_SIGMOID_CROSS_ENTROPY_LOSS_KERNEL(double, int32_t);
REGISTER_SIGMOID_CROSS_ENTROPY_LOSS_KERNEL(double, int64_t);

}  // namespace oneflow
