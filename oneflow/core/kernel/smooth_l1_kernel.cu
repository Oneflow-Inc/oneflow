#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void SmoothL1Forward(const int64_t elem_cnt, const T* prediction, const T* label,
                                const float beta, const float scale, T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T abs_x = std::abs(prediction[i] - label[i]);
    if (abs_x < beta) {
      out[i] = 0.5 * abs_x * abs_x / beta;
    } else {
      out[i] = abs_x - 0.5 * beta;
    }
    out[i] *= scale;
  }
}

template<typename T>
__global__ void SmoothL1Backward(const int64_t elem_cnt, const T* dy, const T* prediction,
                                 const T* label, const float beta, const float scale, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
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

}  // namespace

template<typename T>
class SmoothL1GPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SmoothL1GPUKernel);
  SmoothL1GPUKernel() = default;
  ~SmoothL1GPUKernel() = default;

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
    SmoothL1Forward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(elem_cnt, prediction, label, beta, scale, out);
  }
};

template<typename T>
class SmoothL1GradGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SmoothL1GradGPUKernel);
  SmoothL1GradGPUKernel() = default;
  ~SmoothL1GradGPUKernel() = default;

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
    SmoothL1Backward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(elem_cnt, dy, prediction, label, beta, scale, dx);
  }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSmoothL1Conf, DeviceType::kGPU, float,
                                      SmoothL1GPUKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSmoothL1Conf, DeviceType::kGPU, double,
                                      SmoothL1GPUKernel<double>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSmoothL1GradConf, DeviceType::kGPU, float,
                                      SmoothL1GradGPUKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSmoothL1GradConf, DeviceType::kGPU, double,
                                      SmoothL1GradGPUKernel<double>)

}  // namespace oneflow
