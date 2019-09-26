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
void SmoothL1GradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* dy = BnInOp2Blob("dy");
  const Blob* x = BnInOp2Blob("x");
  const Blob* label = BnInOp2Blob("label");
  Blob* dx_blob = BnInOp2Blob("dx");
  const SmoothL1OpConf& conf = this->op_conf().smooth_l1_conf();
  const float beta = conf.beta();
  const float scale = conf.scale();
  const int64_t elem_cnt = x->shape().elem_cnt();

  SmoothL1KernelUtil<device_type, T>::Backward(ctx.device_ctx, elem_cnt, dy->dptr<T>(),
                                               x->dptr<T>(), label->dptr<T>(), beta, scale,
                                               dx_blob->mut_dptr<T>());
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
  static void Backward(DeviceCtx* ctx, const int64_t elem_cnt, const T* dy, const T* prediction,
                       const T* label, const float beta, const float scale, T* dx) {
    for (int64_t i = 0; i < elem_cnt; i++) {
      const T x = prediction[i] - label[i];
      const T abs_x = std::abs(x);
      if (abs_x < beta) {
        dx[i] = x / beta;
      } else {
        dx[i] = (x > GetZeroVal<T>()) - (x < GetZeroVal<T>());
      }
      dx[i] *= scale * dy[i];
    }
  }
};

#define REGISTER_SMOOTH_L1_KERNEL(dev, dtype)                                        \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSmoothL1Conf, dev, dtype,     \
                                        SmoothL1Kernel<dev, dtype>)                  \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSmoothL1GradConf, dev, dtype, \
                                        SmoothL1GradKernel<dev, dtype>)

REGISTER_SMOOTH_L1_KERNEL(DeviceType::kGPU, float);
REGISTER_SMOOTH_L1_KERNEL(DeviceType::kGPU, double);
REGISTER_SMOOTH_L1_KERNEL(DeviceType::kCPU, float);
REGISTER_SMOOTH_L1_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
