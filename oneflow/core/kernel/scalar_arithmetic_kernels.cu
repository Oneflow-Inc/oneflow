#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<size_t N>
__global__ void HalfPow(const int64_t n, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = __hmul(x[i], x[i]);
#pragma unroll
    for (int i = 2; i < N; ++i) { y[i] = __hmul(y[i], x[i]); }
  }
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
    switch (scalar_operand) {
      case 1: out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob); break;
#define SWITCH_CASE(num)                                                                          \
  case num:                                                                                       \
    HalfPow<num>                                                                                  \
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx.device_ctx->cuda_stream()>>>( \
            n, reinterpret_cast<const half*>(in_blob->dptr<float16>()),                           \
            reinterpret_cast<half*>(out_blob->mut_dptr<float16>()));                              \
    break;
        SWITCH_CASE(2);
        SWITCH_CASE(3);
        SWITCH_CASE(4);
        SWITCH_CASE(5);
        SWITCH_CASE(6);
        SWITCH_CASE(7);
        SWITCH_CASE(8);
#undef SWITCH_CASE
      default: UNIMPLEMENTED();
    }
  }
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().scalar_pow_conf();
  }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kScalarPowConf, DeviceType::kGPU, float16,
                                      ScalarIntPowHalfGpuKernel);

}  // namespace oneflow
