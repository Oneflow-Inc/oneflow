#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

__global__ void HalfPow2(const int64_t n, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = __hmul(x[i], x[i]); }
}

__global__ void HalfPow3(const int64_t n, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = __hmul(x[i], __hmul(x[i], x[i])); }
}

}  // namespace

class ScalarIntPowHalfGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarIntPowHalfGpuKernel);
  ScalarIntPowHalfGpuKernel() = default;
  ~ScalarIntPowHalfGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    int32_t scalar_operand = this->op_conf().scalar_pow_conf().int_operand();
    CHECK_EQ(DataType::kFloat16, in_blob->data_type());
    CHECK_EQ(DataType::kFloat16, out_blob->data_type());
    CHECK_EQ(in_blob->shape(), out_blob->shape());
    int64_t n = in_blob->shape().elem_cnt();
    if (scalar_operand == 1) {
      out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
    } else if (scalar_operand == 2) {
      HalfPow2<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                 ctx.device_ctx->cuda_stream()>>>(
          n, reinterpret_cast<const half*>(in_blob->dptr<float16>()),
          reinterpret_cast<half*>(out_blob->mut_dptr<float16>()));
    } else if (scalar_operand == 3) {
      HalfPow3<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                 ctx.device_ctx->cuda_stream()>>>(
          n, reinterpret_cast<const half*>(in_blob->dptr<float16>()),
          reinterpret_cast<half*>(out_blob->mut_dptr<float16>()));
    } else {
      UNIMPLEMENTED();
    }
  }
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().scalar_pow_conf();
  }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kScalarPowConf, DeviceType::kGPU, float16,
                                      ScalarIntPowHalfGpuKernel);

}  // namespace oneflow
