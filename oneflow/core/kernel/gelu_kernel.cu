#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void GeluForwardGpu(const int64_t n, const T* x, const T inv_sqrt2, T* y) {
  UNIMPLEMENTED();
}

template<typename T>
__global__ void GeluBackwardGpu(const int64_t n, const T* x, const T* dy, const T inv_sqrt2,
                                const T coef, T* dx) {
  UNIMPLEMENTED();
}

template<>
__global__ void GeluForwardGpu(const int64_t n, const float* x, const float inv_sqrt2, float* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 0.5f * x[i] * (1.0f + erff(inv_sqrt2 * x[i])); }
}

template<>
__global__ void GeluBackwardGpu(const int64_t n, const float* x, const float* dy,
                                const float inv_sqrt2, const float coef, float* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    dx[i] =
        0.5f * (1.0f + erff(inv_sqrt2 * x[i]) + x[i] * coef * expf(-0.5f * x[i] * x[i])) * dy[i];
  }
}

template<>
__global__ void GeluForwardGpu(const int64_t n, const double* x, const double inv_sqrt2,
                               double* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 0.5 * x[i] * (1.0 + erf(inv_sqrt2 * x[i])); }
}

template<>
__global__ void GeluBackwardGpu(const int64_t n, const double* x, const double* dy,
                                const double inv_sqrt2, const double coef, double* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    dx[i] = 0.5 * (1.0 + erf(inv_sqrt2 * x[i]) + x[i] * coef * exp(-0.5 * x[i] * x[i])) * dy[i];
  }
}

}  // namespace

template<typename T>
class GeluGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluGpuKernel);
  GeluGpuKernel() = default;
  ~GeluGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    int64_t n = in_blob->static_shape().elem_cnt();
    const T* x = in_blob->dptr<T>();
    T* y = BnInOp2Blob("out")->mut_dptr<T>();

    const T inv_sqrt2 = sqrt(0.5);
    GeluForwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx.device_ctx->cuda_stream()>>>(
            n, x, inv_sqrt2, y);
  }
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().gelu_conf(); }
};

#define REGISTER_GELU_KERNELS(name, dtype)                                                    \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::k##name##Conf, DeviceType::kGPU, dtype, \
                                        name##GpuKernel<dtype>);

REGISTER_GELU_KERNELS(Gelu, float);
REGISTER_GELU_KERNELS(Gelu, double);

template<typename T>
class GeluGradGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluGradGpuKernel);
  GeluGradGpuKernel() = default;
  ~GeluGradGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* x_blob = BnInOp2Blob("x");
    int64_t n = x_blob->static_shape().elem_cnt();
    const T* x = x_blob->dptr<T>();
    const T* dy = BnInOp2Blob("dy")->dptr<T>();
    T* dx = BnInOp2Blob("dx")->mut_dptr<T>();

    const T inv_sqrt2 = sqrt(0.5);
    const T coef = sqrt(2.0 / acos(-1.0));
    GeluBackwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx.device_ctx->cuda_stream()>>>(
            n, x, dy, inv_sqrt2, coef, dx);
  }
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().gelu_grad_conf(); }
};

REGISTER_GELU_KERNELS(GeluGrad, float);
REGISTER_GELU_KERNELS(GeluGrad, double);

#undef REGISTER_GELU_KERNELS

}  // namespace oneflow
