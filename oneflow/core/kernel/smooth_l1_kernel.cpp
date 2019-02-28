#include "oneflow/core/kernel/smooth_l1_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SmoothL1Kernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  const Blob* inside_weights = BnInOp2Blob("inside_weights");
  const Blob* outside_weights = BnInOp2Blob("outside_weights");
  Blob* out = BnInOp2Blob("out");
  Blob* pred_diff_blob = BnInOp2Blob(GenDiffBn("prediction"));
  auto kernel_conf = this->kernel_conf();
  const float beta = kernel_conf.op_attribute().op_conf().smooth_l1_conf().beta();
  const float scale = kernel_conf.op_attribute().op_conf().smooth_l1_conf().scale();
  const int64_t elem_cnt = BnInOp2Blob("prediction")->shape().elem_cnt();

  Memset<device_type>(ctx.device_ctx, out->mut_dptr(), 0, out->ByteSizeOfDataContentField());
  SmoothL1KernelUtil<device_type, T>::Forward(
      ctx.device_ctx, elem_cnt, prediction->dptr<T>(), label->dptr<T>(),
      inside_weights->dptr<T>(), outside_weights->dptr<T>(), beta, scale, out->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void SmoothL1Kernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  const Blob* inside_weights = BnInOp2Blob("inside_weights");
  const Blob* outside_weights = BnInOp2Blob("outside_weights");
  Blob* out = BnInOp2Blob("out");
  Blob* pred_diff_blob = BnInOp2Blob(GenDiffBn("prediction"));
  auto kernel_conf = this->kernel_conf();
  const float beta = kernel_conf.op_attribute().op_conf().smooth_l1_conf().beta();
  const float scale = kernel_conf.op_attribute().op_conf().smooth_l1_conf().scale();
  const int64_t elem_cnt = BnInOp2Blob("prediction")->shape().elem_cnt();

  Memset<device_type>(ctx.device_ctx, pred_diff_blob->mut_dptr(), 0, pred_diff_blob->ByteSizeOfDataContentField());
  SmoothL1KernelUtil<device_type, T>::Backward(
      ctx.device_ctx, elem_cnt, prediction->dptr<T>(), label->dptr<T>(),
      inside_weights->dptr<T>(), outside_weights->dptr<T>(), beta, scale,
      pred_diff_blob->mut_dptr<T>());
}

template<typename T>
struct SmoothL1KernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt,
                      const T* prediction, const T* label, const T* inside_weights,
                      const T* outside_weights, const float beta, const float scale, T* out) {
    for (int i = 0; i < elem_cnt; i++) {
      T x = inside_weights[i] * (prediction[i] - label[i]);
      T abs_x = std::abs(x);
      if (abs_x < beta) {
        out[i] = 0.5 * x * x / beta;
      } else {
        out[i] = abs_x - 0.5 * beta;
      }
      out[i] *= scale * outside_weights[i];
    }
  }
  static void Backward(DeviceCtx* ctx, const int64_t elem_cnt,
                       const T* prediction, const T* label, const T* inside_weights,
                       const T* outside_weights, const float beta, const float scale, T* prediction_diff) {
    for (int i = 0; i < elem_cnt; i++) {
      T x = inside_weights[i] * (prediction[i] - label[i]);
      T abs_x = std::abs(x);
      if (abs_x < beta) {
        prediction_diff[i] = x / beta;
      } else {
        prediction_diff[i] = (x > ZeroVal<T>::value) - (x < ZeroVal<T>::value);
      }
      prediction_diff[i] *= scale * inside_weights[i] * outside_weights[i];
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSmoothL1Conf, SmoothL1Kernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
