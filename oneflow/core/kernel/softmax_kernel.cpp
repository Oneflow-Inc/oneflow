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
  const FloatingPointType* in =
      static_cast<const FloatingPointType*>(in_blob->mut_dptr());
  FloatingPointType* tmp =
      static_cast<FloatingPointType*>(tmp_blob->mut_dptr());
  FloatingPointType* out =
      static_cast<FloatingPointType*>(out_blob->mut_dptr());
  // copy in blob to out blob
  KernelUtil<device_type, FloatingPointType>::BlasCopy(ctx, n * w, in, 1, out,
                                                       1);
  // max | calculate max of every sample vector out[i], store in tmp[i]
  //       the out[i] now is store the data of in[i]
  SoftmaxKernelUtil<device_type, FloatingPointType>::ForwardMax(ctx, n, w, out,
                                                                tmp);
  // sub | every element of out blob subract the max value of the same sample
  for (int64_t i = 0; i < w; ++i) {
    KernelUtil<device_type, FloatingPointType>::BlasAxpy(ctx, n, -1.0, tmp, 1,
                                                         out + i, w);
  }
  // exp | exponentiation every element
  KernelUtil<device_type, FloatingPointType>::Exp(ctx, n * w, out, out);
  // sum | calculate sum of every sample vector out[i], store in tmp[i]
  //       the out[i] now is store the tmp data after exp
  SoftmaxKernelUtil<device_type, FloatingPointType>::ForwardSum(ctx, n, w, out,
                                                                tmp);
  // div | every element of out[i] divided by the data of tmp[i] (the sum value)
  for (int64_t i = 0; i < n; ++i) {
    KernelUtil<device_type, FloatingPointType>::Div(ctx, w, out + i * w,
                                                    tmp + i);
  }
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
  FloatingPointType* in_diff =
      static_cast<FloatingPointType*>(in_diff_blob->mut_dptr());
  FloatingPointType* tmp =
      static_cast<FloatingPointType*>(tmp_blob->mut_dptr());
  const FloatingPointType* out =
      static_cast<const FloatingPointType*>(out_blob->mut_dptr());
  const FloatingPointType* out_diff =
      static_cast<const FloatingPointType*>(out_diff_blob->mut_dptr());
  // copy out_diff to in_diff
  KernelUtil<device_type, FloatingPointType>::BlasCopy(ctx, n * w, out_diff, 1,
                                                       in_diff, 1);
  // dot product | get dot product tmp[i] from out[i] * out_diff[i]
  for (int64_t i = 0; i < n; ++i) {
    KernelUtil<device_type, FloatingPointType>::BlasDot(
        ctx, w, out + i * w, 1, out_diff + i * w, 1, tmp + i);
  }
  // sub | in_diff[i][j] -= tmp[i]
  for (int64_t i = 0; i < w; ++i) {
    KernelUtil<device_type, FloatingPointType>::BlasAxpy(ctx, n, -1.0, tmp, 1,
                                                         in_diff + i, w);
  }
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
                                                           tmp + i);
    }
  }

  static void ForwardSum(const KernelCtx& ctx, const int64_t n, const int64_t w,
                         const FloatingPointType* out, FloatingPointType* tmp) {
    for (int64_t i = 0; i < n; ++i) {
      KernelUtil<DeviceType::kCPU, FloatingPointType>::Sum(ctx, w, out + i * w,
                                                           tmp + i);
    }
  }
};

INSTANTIATE_CPU_KERNEL_UTIL_CLASS(SoftmaxKernelUtil);
INSTANTIATE_KERNEL_CLASS(SoftmaxKernel);
REGISTER_KERNEL(OperatorConf::kSoftmaxConf, SoftmaxKernel);

}  // namespace oneflow
