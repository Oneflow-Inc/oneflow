#include "oneflow/core/kernel/smooth_l1_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SmoothL1Kernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  Blob* out = BnInOp2Blob("out");
  const SmoothL1OpConf& conf = this->op_conf().smooth_l1_conf();
  const float beta = conf.beta();
  const float scale = conf.scale();
  const int64_t elem_cnt = prediction->shape().elem_cnt();

  SmoothL1KernelUtil<device_type, T>::Forward(ctx.device_ctx, elem_cnt, prediction->dptr<T>(),
                                              label->dptr<T>(), beta, scale, out->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void SmoothL1Kernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff = BnInOp2Blob(GenDiffBn("out"));
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  Blob* prediction_diff_blob = BnInOp2Blob(GenDiffBn("prediction"));
  const SmoothL1OpConf& conf = this->op_conf().smooth_l1_conf();
  const float beta = conf.beta();
  const float scale = conf.scale();
  const int64_t elem_cnt = prediction->shape().elem_cnt();

  SmoothL1KernelUtil<device_type, T>::Backward(ctx.device_ctx, elem_cnt, out_diff->dptr<T>(),
                                               prediction->dptr<T>(), label->dptr<T>(), beta, scale,
                                               prediction_diff_blob->mut_dptr<T>());
}

template<typename T>
struct SmoothL1KernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const T* prediction, const T* label,
                      const float beta, const float scale, T* out) {
    for (int64_t i = 0; i < elem_cnt; i++) {
      const T abs_x = std::abs(prediction[i] - label[i]);
      if (abs_x < beta) {
        out[i] = 0.5 * abs_x * abs_x / beta;
      } else {
        out[i] = abs_x - 0.5 * beta;
      }
      out[i] *= scale;
    }
  }
  static void Backward(DeviceCtx* ctx, const int64_t elem_cnt, const T* out_diff,
                       const T* prediction, const T* label, const float beta, const float scale,
                       T* prediction_diff) {
    for (int64_t i = 0; i < elem_cnt; i++) {
      const T x = prediction[i] - label[i];
      const T abs_x = std::abs(x);
      if (abs_x < beta) {
        prediction_diff[i] = x / beta;
      } else {
        prediction_diff[i] = (x > ZeroVal<T>::value) - (x < ZeroVal<T>::value);
      }
      prediction_diff[i] *= scale * out_diff[i];
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSmoothL1Conf, SmoothL1Kernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
