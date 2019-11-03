#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename K>
__device__ int64_t GetOffset(const int64_t batch_idx, const int64_t num_classes,
                             const int64_t lower_bound, const K* label) {
  const int64_t idx = label[batch_idx] - lower_bound;
  if (idx >= 0 && idx < num_classes) {
    return batch_idx * num_classes + idx;
  } else {
    return -1;
  }
}

template<typename T, typename K>
__global__ void GpuForward(const int64_t num_instances, const int64_t num_classes,
                           const int64_t lower_bound, const T cos_m, const T sin_m, const T* in,
                           const K* label, T* sin_theta_data, T* out) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    const int64_t idx = GetOffset<K>(i, num_classes, lower_bound, label);
    if (idx != -1) {
      sin_theta_data[i] = sqrt(1 - in[idx] * in[idx]);
      out[idx] = in[idx] * cos_m - sin_theta_data[i] * sin_m;
      sin_theta_data[i] = in[idx] / sin_theta_data[i];
    }
  }
}

template<typename T, typename K>
__global__ void GpuBackward(const int64_t num_instances, const int64_t num_classes,
                            const int64_t lower_bound, const T cos_m, const T sin_m,
                            const T* out_diff, const K* label, const T* sin_theta_data,
                            T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    const int64_t idx = GetOffset<K>(i, num_classes, lower_bound, label);
    if (idx != -1) { in_diff[idx] = in_diff[idx] * (1 * cos_m + sin_theta_data[i] * sin_m); }
  }
}

}  // namespace

template<typename T, typename K>
class AdditiveAngularMarginGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdditiveAngularMarginGpuKernel);
  AdditiveAngularMarginGpuKernel() = default;
  ~AdditiveAngularMarginGpuKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override {
    if (this->op_conf().has_additive_angular_margin_conf()) {
      return this->op_conf().additive_angular_margin_conf();
    } else if (this->op_conf().has_additive_angular_margin_ms1_conf()) {
      return this->op_conf().additive_angular_margin_ms1_conf();
    } else {
      UNIMPLEMENTED();
    }
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    const Blob* label_blob = BnInOp2Blob("label");
    Blob* sin_theta_data_blob = BnInOp2Blob("sin_theta_data");
    Blob* out_blob = BnInOp2Blob("out");
    const float margin = GetValFromPbMessage<float>(this->GetCustomizedOpConf(), "margin");
    int64_t lower_bound = 0;
    if (this->kernel_conf().additive_angular_margin_conf().has_lower_bound()) {
      lower_bound = this->kernel_conf().additive_angular_margin_conf().lower_bound();
    }
    out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
    Memset<DeviceType::kGPU>(ctx.device_ctx, sin_theta_data_blob->mut_dptr(), 0,
                             BnInOp2Blob("sin_theta_data")->ByteSizeOfDataContentField());
    GpuForward<<<BlocksNum4ThreadsNum(in_blob->shape().At(0)), kCudaThreadsNumPerBlock, 0,
                 ctx.device_ctx->cuda_stream()>>>(
        in_blob->shape().At(0), in_blob->shape().Count(1), lower_bound, static_cast<T>(cos(margin)),
        static_cast<T>(sin(margin)), in_blob->dptr<T>(), label_blob->dptr<K>(),
        sin_theta_data_blob->mut_dptr<T>(), out_blob->mut_dptr<T>());
  }
};

template<typename T, typename K>
class AdditiveAngularMarginGradGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdditiveAngularMarginGradGPUKernel);
  AdditiveAngularMarginGradGPUKernel() = default;
  ~AdditiveAngularMarginGradGPUKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override {
    if (this->op_conf().has_additive_angular_margin_grad_conf()) {
      return this->op_conf().additive_angular_margin_grad_conf();
    } else if (this->op_conf().has_additive_angular_margin_ms1_grad_conf()) {
      return this->op_conf().additive_angular_margin_ms1_grad_conf();
    } else {
      UNIMPLEMENTED();
    }
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* dy_blob = BnInOp2Blob("dy");
    const Blob* label_blob = BnInOp2Blob("label");
    Blob* sin_theta_data_blob = BnInOp2Blob("sin_theta_data");
    Blob* dx_blob = BnInOp2Blob("dx");
    const float margin = GetValFromPbMessage<float>(this->GetCustomizedOpConf(), "margin");
    int64_t lower_bound = 0;
    if (this->kernel_conf().has_additive_angular_margin_grad_conf()) {
      lower_bound = this->kernel_conf().additive_angular_margin_grad_conf().lower_bound();
    }
    dx_blob->CopyDataContentFrom(ctx.device_ctx, dy_blob);
    GpuBackward<<<BlocksNum4ThreadsNum(dy_blob->shape().At(0)), kCudaThreadsNumPerBlock, 0,
                  ctx.device_ctx->cuda_stream()>>>(
        dy_blob->shape().At(0), dy_blob->shape().Count(1), lower_bound, static_cast<T>(cos(margin)),
        static_cast<T>(sin(margin)), dy_blob->dptr<T>(), label_blob->dptr<K>(),
        sin_theta_data_blob->dptr<T>(), dx_blob->mut_dptr<T>());
  }
};

#define REGISTER_ADDITIVE_ANGULAR_MARGIN_AND_GRAD_GPU_KERNEL(dtype, ltype)                       \
  NEW_REGISTER_KERNEL(OperatorConf::kAdditiveAngularMarginConf,                                  \
                      AdditiveAngularMarginGpuKernel<dtype, ltype>)                              \
      .SetIsMatchedPred([](const KernelConf& conf) {                                             \
        return (                                                                                 \
            (conf.op_attribute().op_conf().device_type() == DeviceType::kGPU)                    \
            && (GetDataType<dtype>::value == conf.data_type())                                   \
            && (GetDataType<ltype>::value == conf.additive_angular_margin_conf().label_type())); \
      });                                                                                        \
  NEW_REGISTER_KERNEL(OperatorConf::kAdditiveAngularMarginMs1Conf,                               \
                      AdditiveAngularMarginGpuKernel<dtype, ltype>)                              \
      .SetIsMatchedPred([](const KernelConf& conf) {                                             \
        return (                                                                                 \
            (conf.op_attribute().op_conf().device_type() == DeviceType::kGPU)                    \
            && (GetDataType<dtype>::value == conf.data_type())                                   \
            && (GetDataType<ltype>::value == conf.additive_angular_margin_conf().label_type())); \
      });                                                                                        \
  NEW_REGISTER_KERNEL(OperatorConf::kAdditiveAngularMarginGradConf,                              \
                      AdditiveAngularMarginGradGPUKernel<dtype, ltype>)                          \
      .SetIsMatchedPred([](const KernelConf& conf) {                                             \
        return ((conf.op_attribute().op_conf().device_type() == DeviceType::kGPU)                \
                && (GetDataType<dtype>::value == conf.data_type())                               \
                && (GetDataType<ltype>::value                                                    \
                    == conf.additive_angular_margin_grad_conf().label_type()));                  \
      });                                                                                        \
  NEW_REGISTER_KERNEL(OperatorConf::kAdditiveAngularMarginMs1GradConf,                           \
                      AdditiveAngularMarginGradGPUKernel<dtype, ltype>)                          \
      .SetIsMatchedPred([](const KernelConf& conf) {                                             \
        return ((conf.op_attribute().op_conf().device_type() == DeviceType::kGPU)                \
                && (GetDataType<dtype>::value == conf.data_type())                               \
                && (GetDataType<ltype>::value                                                    \
                    == conf.additive_angular_margin_grad_conf().label_type()));                  \
      });

REGISTER_ADDITIVE_ANGULAR_MARGIN_AND_GRAD_GPU_KERNEL(float, int64_t);
REGISTER_ADDITIVE_ANGULAR_MARGIN_AND_GRAD_GPU_KERNEL(double, int64_t);
REGISTER_ADDITIVE_ANGULAR_MARGIN_AND_GRAD_GPU_KERNEL(float, int32_t);
REGISTER_ADDITIVE_ANGULAR_MARGIN_AND_GRAD_GPU_KERNEL(double, int32_t);
REGISTER_ADDITIVE_ANGULAR_MARGIN_AND_GRAD_GPU_KERNEL(float, int8_t);
REGISTER_ADDITIVE_ANGULAR_MARGIN_AND_GRAD_GPU_KERNEL(double, int8_t);

}  // namespace oneflow