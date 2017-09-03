#include "oneflow/core/kernel/softmax_loss_kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void SoftmaxLossKernel<device_type, PredType, LabelType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* prediction_blob = BnInOp2BlobPtr("prediction");
  const Blob* label_blob = BnInOp2BlobPtr("label");
  Blob* prob_blob = BnInOp2BlobPtr("prob");
  Blob* tmp_blob = BnInOp2BlobPtr("tmp_1D");
  Blob* loss_blob = BnInOp2BlobPtr("loss");
  const int64_t n = prediction_blob->shape().At(0);
  const int64_t w = prediction_blob->shape().Count(1);
  const PredType* in = prediction_blob->dptr<PredType>();
  const LabelType* label = label_blob->dptr<LabelType>();
  PredType* tmp = tmp_blob->mut_dptr<PredType>();
  PredType* prob = prob_blob->mut_dptr<PredType>();
  PredType* loss = loss_blob->mut_dptr<PredType>();
  // forward
  SoftmaxComputeProb<device_type, PredType>(ctx.device_ctx, n, w, in, tmp,
                                            prob);
  SoftmaxLossKernelUtil<device_type, PredType, LabelType>::ComputeLoss(
      ctx.device_ctx, n, w, label, prob, tmp, loss);
  // backward
  // if prediction_diff_blob is not null , then do backward
  Blob* prediction_diff_blob = BnInOp2BlobPtr(GenDiffBn("prediction"));
  if (prediction_diff_blob != nullptr) {
    PredType* in_diff = prediction_diff_blob->mut_dptr<PredType>();
    KernelUtil<device_type, PredType>::BlasCopy(ctx.device_ctx, n * w, prob, 1,
                                                in_diff, 1);
    SoftmaxLossKernelUtil<device_type, PredType, LabelType>::BackwardSub(
        ctx.device_ctx, n, w, label, in_diff);
  }
}

template<typename PredType, typename LabelType>
class SoftmaxLossKernelUtil<DeviceType::kCPU, PredType, LabelType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxLossKernelUtil);
  SoftmaxLossKernelUtil() = delete;

  static void ComputeLoss(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const LabelType* label, const PredType* prob,
                          PredType* tmp, PredType* loss) {
    ctx->cpu_stream()->SendWork([=]() {
      *loss = 0;
      for (int64_t i = 0; i < n; ++i) {
        *loss -= SAFE_LOG(prob[i * w + static_cast<int64_t>(label[i])]);
      }
    });
  }

  static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const LabelType* label, PredType* in_diff) {
    ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        in_diff[i * w + static_cast<int64_t>(label[i])] -= 1;
      }
    });
  }
};

}  // namespace oneflow
