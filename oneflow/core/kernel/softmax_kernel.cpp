#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void SoftmaxKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_blob = BnInOp2BlobPtr(op()->SoleIbn());
  Blob* out_blob = BnInOp2BlobPtr(op()->SoleObn());
  Blob* tmp_blob = BnInOp2BlobPtr(op()->SoleDtbn());
  const int64_t n = out_blob->shape().At(0);
  const int64_t w = out_blob->shape().At(1);
  const FloatingPointType* in = in_blob->dptr<FloatingPointType>();
  FloatingPointType* tmp = tmp_blob->mut_dptr<FloatingPointType>();
  FloatingPointType* out = out_blob->mut_dptr<FloatingPointType>();
  SoftmaxComputeProb<device_type, FloatingPointType>(ctx, n, w, in, tmp, out);
}

template<DeviceType device_type, typename FloatingPointType>
void SoftmaxKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* out_blob = BnInOp2BlobPtr(op()->SoleObn());
  Blob* out_diff_blob = BnInOp2BlobPtr(op()->SoleOdbn());
  Blob* in_diff_blob = BnInOp2BlobPtr(op()->SoleIdbn());
  Blob* tmp_blob = BnInOp2BlobPtr(op()->SoleDtbn());
  const int64_t n = out_blob->shape().At(0);
  const int64_t w = out_blob->shape().At(1);
  FloatingPointType* in_diff = in_diff_blob->mut_dptr<FloatingPointType>();
  FloatingPointType* tmp = tmp_blob->mut_dptr<FloatingPointType>();
  const FloatingPointType* out = out_blob->dptr<FloatingPointType>();
  const FloatingPointType* out_diff = out_diff_blob->dptr<FloatingPointType>();
  // copy out_diff to in_diff
  KernelUtil<device_type, FloatingPointType>::BlasCopy(ctx, n * w, out_diff, 1,
                                                       in_diff, 1);
  // dot product | get dot product tmp[i] from out[i] * out_diff[i]
  SoftmaxKernelUtil<device_type, FloatingPointType>::BackwardDot(ctx, n, w, out,
                                                                 out_diff, tmp);
  // sub | in_diff[i][j] -= tmp[i]
  SoftmaxKernelUtil<device_type, FloatingPointType>::Sub(ctx, n, w, in_diff,
                                                         tmp);
  // elementwise multiplication | in_diff[i][j] *= out[i][j]
  KernelUtil<device_type, FloatingPointType>::Mul(ctx, n * w, in_diff, out,
                                                  in_diff);
}

template<typename FloatingPointType>
class SoftmaxKernelUtil<DeviceType::kCPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxKernelUtil);
  SoftmaxKernelUtil() = delete;

  static void ForwardMax(const KernelCtx& ctx, const int64_t n, const int64_t w,
                         const FloatingPointType* out, FloatingPointType* tmp) {
    for (int64_t i = 0; i < n; ++i) {
      KernelUtil<DeviceType::kCPU, FloatingPointType>::Max(ctx, w, out + i * w,
                                                           tmp + i, nullptr, 0);
    }
  }

  static void ForwardSum(const KernelCtx& ctx, const int64_t n, const int64_t w,
                         const FloatingPointType* out, FloatingPointType* tmp) {
    for (int64_t i = 0; i < n; ++i) {
      KernelUtil<DeviceType::kCPU, FloatingPointType>::Sum(ctx, w, out + i * w,
                                                           tmp + i, nullptr, 0);
    }
  }

  static void Sub(const KernelCtx& ctx, const int64_t n, const int64_t w,
                  FloatingPointType* matrix, const FloatingPointType* vector) {
    for (int64_t i = 0; i < w; ++i) {
      KernelUtil<DeviceType::kCPU, FloatingPointType>::BlasAxpy(
          ctx, n, static_cast<FloatingPointType>(-1.0), vector, 1, matrix + i,
          w);
    }
  }

  static void BackwardDot(const KernelCtx& ctx, const int64_t n,
                          const int64_t w, const FloatingPointType* out,
                          const FloatingPointType* out_diff,
                          FloatingPointType* tmp) {
    for (int64_t i = 0; i < n; ++i) {
      KernelUtil<DeviceType::kCPU, FloatingPointType>::BlasDot(
          ctx, w, out + i * w, 1, out_diff + i * w, 1, tmp + i);
    }
  }
};

INSTANTIATE_CPU_KERNEL_UTIL_CLASS(SoftmaxKernelUtil);
INSTANTIATE_KERNEL_CLASS(SoftmaxKernel);
REGISTER_KERNEL(OperatorConf::kSoftmaxConf, SoftmaxKernel);

}  // namespace oneflow
