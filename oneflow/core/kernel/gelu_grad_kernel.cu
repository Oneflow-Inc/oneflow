#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

__global__ void GeluBackwardGpuHalf(const int64_t n, const half* x, const half* dy,
                                    const float inv_sqrt2, const float coef, half* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    float f_x = __half2float(x[i]);
    dx[i] =
        __float2half(0.5f * (1.0f + erff(inv_sqrt2 * f_x) + f_x * coef * expf(-0.5f * f_x * f_x))
                     * __half2float(dy[i]));
  }
}

}  // namespace

class GeluGradGpuHalfKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluGradGpuHalfKernel);
  GeluGradGpuHalfKernel() = default;
  ~GeluGradGpuHalfKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* x_blob = BnInOp2Blob("x");
    const int64_t n = x_blob->static_shape().elem_cnt();
    const float inv_sqrt2 = sqrt(0.5);
    const float coef = sqrt(2.0 / acos(-1.0));
    GeluBackwardGpuHalf<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                          ctx.device_ctx->cuda_stream()>>>(
        n, reinterpret_cast<const half*>(x_blob->dptr<float16>()),
        reinterpret_cast<const half*>(BnInOp2Blob("dy")->dptr<float16>()), inv_sqrt2, coef,
        reinterpret_cast<half*>(BnInOp2Blob("dx")->mut_dptr<float16>()));
  }

  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().gelu_grad_conf(); }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kGeluGradConf, DeviceType::kGPU, float16,
                                      GeluGradGpuHalfKernel)

}  // namespace oneflow
