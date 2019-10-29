#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"

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

__inline__ __device__ half zero_point_five() { return __float2half(0.5); }
__inline__ __device__ half magic_number() { return __float2half(0.044715); }
__inline__ __device__ half magic_number_times_three() { return __float2half(0.044715 * 3); }
__inline__ __device__ half sqrt_two_divide_pi() { return __float2half(0.79788456080286541); }
__inline__ __device__ half square(const half x) { return __hmul(x, x); }

__inline__ __device__ half Tanh(const half x) {
  half ex = hexp(x);
  half e_x = hexp(__hneg(x));
  return __hdiv(__hsub(ex, e_x), __hadd(ex, e_x));
}

__inline__ __device__ half SimpleFunc(const half x) {
  half cub = __hmul(square(x), x);
  half tmp = __hadd(x, __hmul(magic_number(), cub));
  return __hmul(sqrt_two_divide_pi(), tmp);  // magic number is sqrt(2/pi)
}

// See gelu() in official bert/modeling.py for more details
__global__ void GeluForwardHalfGpu(const int64_t n, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = __hmul(zero_point_five(), __hadd(hone(), Tanh(SimpleFunc(x[i]))));
  }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

__inline__ __device__ half TanhBackward(const half dy, const half y) {
  return __hmul(dy, __hsub(hone(), __hmul(y, y)));
}

__global__ void GeluBackwardHalfGpu(const int64_t n, const half* x, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) {
    half tmp_x = x[i];
    half y = Tanh(SimpleFunc(tmp_x));
    half tmp_dy = __hmul(zero_point_five(), dy[i]);
    tmp_dy = TanhBackward(tmp_dy, y);
    tmp_dy = __hmul(sqrt_two_divide_pi(), tmp_dy);
    half tmp = __hadd(hone(), __hmul(magic_number_times_three(), square(tmp_x)));
    dx[i] = __hmul(tmp_dy, tmp);
  }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
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

class GeluHalfGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluHalfGpuKernel);
  GeluHalfGpuKernel() = default;
  ~GeluHalfGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    int64_t n = in_blob->static_shape().elem_cnt();
    const float16* x = in_blob->dptr<float16>();
    float16* y = BnInOp2Blob("out")->mut_dptr<float16>();

    GeluForwardHalfGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                         ctx.device_ctx->cuda_stream()>>>(n, reinterpret_cast<const half*>(x),
                                                          reinterpret_cast<half*>(y));
  }
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().gelu_conf(); }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kGeluConf, DeviceType::kGPU, float16,
                                      GeluHalfGpuKernel);

class GeluGradHalfGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluGradHalfGpuKernel);
  GeluGradHalfGpuKernel() = default;
  ~GeluGradHalfGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* x_blob = BnInOp2Blob("x");
    int64_t n = x_blob->static_shape().elem_cnt();
    const float16* x = x_blob->dptr<float16>();
    const float16* dy = BnInOp2Blob("dy")->dptr<float16>();
    float16* dx = BnInOp2Blob("dx")->mut_dptr<float16>();

    GeluBackwardHalfGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                          ctx.device_ctx->cuda_stream()>>>(n, reinterpret_cast<const half*>(x),
                                                           reinterpret_cast<const half*>(dy),
                                                           reinterpret_cast<half*>(dx));
  }
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().gelu_grad_conf(); }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kGeluGradConf, DeviceType::kGPU, float16,
                                      GeluGradHalfGpuKernel);

}  // namespace oneflow
