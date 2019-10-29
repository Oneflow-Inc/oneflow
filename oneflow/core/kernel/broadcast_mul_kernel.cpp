#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<typename T>
class BroadcastMulCpuKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMulCpuKernel);
  BroadcastMulCpuKernel() = default;
  ~BroadcastMulCpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* a = BnInOp2Blob("a");
    const Blob* b = BnInOp2Blob("b");
    Blob* out = BnInOp2Blob("out");
    int64_t n = out->shape().elem_cnt();
    if (a->shape().elem_cnt() == 1) {
      CHECK_EQ(n, b->shape().elem_cnt());
      NewKernelUtil<DeviceType::kCPU>::MulByScalar(ctx.device_ctx, n, b->dptr<T>(), *(a->dptr<T>()),
                                                   out->mut_dptr<T>());
    } else if (b->shape().elem_cnt() == 1) {
      CHECK_EQ(n, a->shape().elem_cnt());
      NewKernelUtil<DeviceType::kCPU>::MulByScalar(ctx.device_ctx, n, a->dptr<T>(), *(b->dptr<T>()),
                                                   out->mut_dptr<T>());
    } else {
      size_t num_axes = out->shape().NumAxes();
      NdarrayUtil<DeviceType::kCPU, T>::BroadcastMul(
          ctx.device_ctx, XpuVarNdarray<T>(out, num_axes), XpuVarNdarray<const T>(a, num_axes),
          XpuVarNdarray<const T>(b, num_axes));
    }
  }
};

template<typename T>
class BroadcastMulGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMulGpuKernel);
  BroadcastMulGpuKernel() = default;
  ~BroadcastMulGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* a = BnInOp2Blob("a");
    const Blob* b = BnInOp2Blob("b");
    Blob* out = BnInOp2Blob("out");
    int64_t n = out->shape().elem_cnt();
    if (a->shape().elem_cnt() == 1) {
      CHECK_EQ(n, b->shape().elem_cnt());
      NewKernelUtil<DeviceType::kGPU>::MulByGpuScalar(ctx.device_ctx, n, b->dptr<T>(), a->dptr<T>(),
                                                      out->mut_dptr<T>());
    } else if (b->shape().elem_cnt() == 1) {
      CHECK_EQ(n, a->shape().elem_cnt());
      NewKernelUtil<DeviceType::kGPU>::MulByGpuScalar(ctx.device_ctx, n, a->dptr<T>(), b->dptr<T>(),
                                                      out->mut_dptr<T>());
    } else {
      size_t num_axes = out->shape().NumAxes();
      NdarrayUtil<DeviceType::kGPU, T>::BroadcastMul(
          ctx.device_ctx, XpuVarNdarray<T>(out, num_axes), XpuVarNdarray<const T>(a, num_axes),
          XpuVarNdarray<const T>(b, num_axes));
    }
  }
};

#define REGISTER_BROADCAST_MUL_KERNEL(dtype)                                                      \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcastMulConf, DeviceType::kCPU, dtype, \
                                        BroadcastMulCpuKernel<dtype>)                             \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcastMulConf, DeviceType::kGPU, dtype, \
                                        BroadcastMulGpuKernel<dtype>)

REGISTER_BROADCAST_MUL_KERNEL(float);
REGISTER_BROADCAST_MUL_KERNEL(double);
REGISTER_BROADCAST_MUL_KERNEL(int32_t);
REGISTER_BROADCAST_MUL_KERNEL(int64_t);

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcastMulConf, DeviceType::kGPU, float16,
                                      BroadcastMulGpuKernel<float16>);

#undef REGISTER_BROADCAST_MUL_KERNEL

}  // namespace oneflow
