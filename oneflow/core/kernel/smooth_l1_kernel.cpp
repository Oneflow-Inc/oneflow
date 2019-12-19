#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class SmoothL1CPUKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SmoothL1CPUKernel);
  SmoothL1CPUKernel() = default;
  ~SmoothL1CPUKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const SmoothL1OpConf& conf = this->op_conf().smooth_l1_conf();
    const float beta = conf.beta();
    const float scale = conf.scale();
    const Blob* prediction_blob = BnInOp2Blob("prediction");
    const T* prediction = prediction_blob->dptr<T>();
    const int64_t elem_cnt = prediction_blob->shape().elem_cnt();
    const T* label = BnInOp2Blob("label")->dptr<T>();
    T* out = BnInOp2Blob("out")->mut_dptr<T>();
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
};

template<typename T>
class SmoothL1GradCPUKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SmoothL1GradCPUKernel);
  SmoothL1GradCPUKernel() = default;
  ~SmoothL1GradCPUKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const SmoothL1GradOpConf& conf = this->op_conf().smooth_l1_grad_conf();
    const float beta = conf.beta();
    const float scale = conf.scale();
    const Blob* x_blob = BnInOp2Blob("x");
    const T* prediction = x_blob->dptr<T>();
    const int64_t elem_cnt = x_blob->shape().elem_cnt();
    const T* dy = BnInOp2Blob("dy")->dptr<T>();
    const T* label = BnInOp2Blob("label")->dptr<T>();
    T* dx = BnInOp2Blob("dx")->mut_dptr<T>();
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

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSmoothL1Conf, DeviceType::kCPU, float,
                                      SmoothL1CPUKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSmoothL1Conf, DeviceType::kCPU, double,
                                      SmoothL1CPUKernel<double>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSmoothL1GradConf, DeviceType::kCPU, float,
                                      SmoothL1GradCPUKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSmoothL1GradConf, DeviceType::kCPU, double,
                                      SmoothL1GradCPUKernel<double>)

}  // namespace oneflow
