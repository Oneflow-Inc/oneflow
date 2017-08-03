#include "oneflow/core/kernel/softmax_loss_kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void SoftmaxLossKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_blob = BnInOp2BlobPtr("in");
  Blob* label_blob = BnInOp2BlobPtr("label");
  Blob* prob_blob = BnInOp2BlobPtr("prob");
  Blob* tmp_blob = BnInOp2BlobPtr("tmp_1D");
  Blob* loss_blob = BnInOp2BlobPtr("loss");
  const int64_t n = in_blob->shape().At(0);
  const int64_t w = in_blob->shape().At(1);
  const FloatingPointType* in = in_blob->dptr<FloatingPointType>();
  const FloatingPointType* label = label_blob->dptr<FloatingPointType>();
  FloatingPointType* tmp = tmp_blob->mut_dptr<FloatingPointType>();
  FloatingPointType* prob = prob_blob->mut_dptr<FloatingPointType>();
  FloatingPointType* loss = loss_blob->mut_dptr<FloatingPointType>();
  // forward
  SoftmaxComputeProb<device_type, FloatingPointType>(ctx, n, w, in, tmp, prob);
  SoftmaxLossKernelUtil<device_type, FloatingPointType>::ComputeLoss(
      ctx, n, w, label, prob, tmp, loss);
  // backward
  // if in_diff_blob is not null , then do backward
  Blob* in_diff_blob = BnInOp2BlobPtr("in_diff");
  if (in_diff_blob != nullptr) {
    FloatingPointType* in_diff = in_diff_blob->mut_dptr<FloatingPointType>();
    KernelUtil<device_type, FloatingPointType>::BlasCopy(ctx, n * w, prob, 1,
                                                         in_diff, 1);
    SoftmaxLossKernelUtil<device_type, FloatingPointType>::BackwardSub(
        ctx, n, w, label, in_diff);
  }
}

template<typename FloatingPointType>
class SoftmaxLossKernelUtil<DeviceType::kCPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxLossKernelUtil);
  SoftmaxLossKernelUtil() = delete;

  static void ComputeLoss(const KernelCtx& ctx, const int64_t n,
                          const int64_t w, const FloatingPointType* label,
                          const FloatingPointType* prob, FloatingPointType* tmp,
                          FloatingPointType* loss) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      *loss = 0;
      for (int64_t i = 0; i < n; ++i) {
        *loss -= SAFE_LOG(prob[i * w + static_cast<int64_t>(label[i])]);
      }
    });
  }

  static void BackwardSub(const KernelCtx& ctx, const int64_t n,
                          const int64_t w, const FloatingPointType* label,
                          FloatingPointType* in_diff) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        in_diff[i * w + static_cast<int64_t>(label[i])] -= 1;
      }
    });
  }
};

INSTANTIATE_CPU_KERNEL_UTIL_CLASS(SoftmaxLossKernelUtil);
INSTANTIATE_KERNEL_CLASS(SoftmaxLossKernel);
REGISTER_KERNEL(OperatorConf::kSoftmaxLossConf, SoftmaxLossKernel);

}  // namespace oneflow
