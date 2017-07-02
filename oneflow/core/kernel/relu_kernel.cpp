#include "oneflow/core/kernel/relu_kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void ReluKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* in_data = BnInOp2BlobPtr("in");
  Blob* out_data = BnInOp2BlobPtr("out");
  ReluKernelUtil<device_type, FloatingPointType>::Forward(
      ctx, out_data->shape().elem_cnt(),
      static_cast<const FloatingPointType*>(in_data->dptr()),
      static_cast<FloatingPointType*>(out_data->mut_dptr()));
}

template<DeviceType device_type, typename FloatingPointType>
void ReluKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_data = BnInOp2BlobPtr("in");
  Blob* out_diff = BnInOp2BlobPtr("out_diff");
  Blob* in_diff = BnInOp2BlobPtr("in_diff");
  ReluKernelUtil<device_type, FloatingPointType>::Backward(
      ctx, in_data->shape().elem_cnt(),
      static_cast<const FloatingPointType*>(out_diff->dptr()),
      static_cast<const FloatingPointType*>(in_data->dptr()),
      static_cast<FloatingPointType*>(in_diff->mut_dptr()));
}

template<typename FloatingPointType>
class ReluKernelUtil<DeviceType::kCPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluKernelUtil);
  ReluKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t n,
                      const FloatingPointType* in, FloatingPointType* out) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        out[i] = std::max(in[i], static_cast<FloatingPointType>(0.0));
      }
    });
  }

  static void Backward(const KernelCtx& ctx, const int64_t n,
                       const FloatingPointType* out_diff,
                       const FloatingPointType* in,
                       FloatingPointType* in_diff) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        in_diff[i] = in[i] > 0 ? out_diff[i] : 0;
      }
    });
  }
};

INSTANTIATE_CPU_KERNEL_UTIL_CLASS(ReluKernelUtil);
INSTANTIATE_KERNEL_CLASS(ReluKernel);
REGISTER_KERNEL(OperatorConf::kReluConf, ReluKernel);

}  // namespace oneflow
