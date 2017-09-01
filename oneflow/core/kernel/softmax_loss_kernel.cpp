#include "oneflow/core/kernel/softmax_loss_kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename LabelType>
void SoftmaxLossKernel<device_type, T, LabelType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* prediction_blob = BnInOp2BlobPtr("prediction");
  const Blob* label_blob = BnInOp2BlobPtr("label");
  Blob* prob_blob = BnInOp2BlobPtr("prob");
  Blob* tmp_blob = BnInOp2BlobPtr("tmp_1D");
  Blob* loss_blob = BnInOp2BlobPtr("loss");
  const int64_t n = prediction_blob->shape().At(0);
  const int64_t w = prediction_blob->shape().Count(1);
  const T* in = prediction_blob->dptr<T>();
  const LabelType* label = label_blob->dptr<LabelType>();
  T* tmp = tmp_blob->mut_dptr<T>();
  T* prob = prob_blob->mut_dptr<T>();
  T* loss = loss_blob->mut_dptr<T>();
  // forward
  SoftmaxComputeProb<device_type, T>(ctx.device_ctx, n, w, in, tmp, prob);
  SoftmaxLossKernelUtil<device_type, T, LabelType>::ComputeLoss(
      ctx.device_ctx, n, w, label, prob, tmp, loss);
  // backward
  // if prediction_diff_blob is not null , then do backward
  Blob* prediction_diff_blob = BnInOp2BlobPtr(GenDiffBn("prediction"));
  if (prediction_diff_blob != nullptr) {
    T* in_diff = prediction_diff_blob->mut_dptr<T>();
    KernelUtil<device_type, T>::BlasCopy(ctx.device_ctx, n * w, prob, 1,
                                         in_diff, 1);
    SoftmaxLossKernelUtil<device_type, T, LabelType>::BackwardSub(
        ctx.device_ctx, n, w, label, in_diff);
  }
}

template<typename T, typename LabelType>
class SoftmaxLossKernelUtil<DeviceType::kCPU, T, LabelType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxLossKernelUtil);
  SoftmaxLossKernelUtil() = delete;

  static void ComputeLoss(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const LabelType* label, const T* prob, T* tmp,
                          T* loss) {
    ctx->cpu_stream()->SendWork([=]() {
      *loss = 0;
      for (int64_t i = 0; i < n; ++i) {
        *loss -= SAFE_LOG(prob[i * w + static_cast<int64_t>(label[i])]);
      }
    });
  }

  static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const LabelType* label, T* in_diff) {
    ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        in_diff[i * w + static_cast<int64_t>(label[i])] -= 1;
      }
    });
  }
};

namespace {

template<DeviceType device_type>
Kernel* CreateSoftmaxLossKernel(const OperatorConf& op_conf) {
  static const HashMap<std::string, std::function<Kernel*()>>
      data_type2creator = {
#define SOFTMAX_LOSS_KERNEL_ENTRY(data_type_pair, label_type_pair)    \
  {GetHashKey(OF_PP_PAIR_SECOND(data_type_pair),                      \
              OF_PP_PAIR_SECOND(label_type_pair)),                    \
   []() {                                                             \
     return new SoftmaxLossKernel<device_type,                        \
                                  OF_PP_PAIR_FIRST(data_type_pair),   \
                                  OF_PP_PAIR_FIRST(label_type_pair)>; \
   }},
          OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SOFTMAX_LOSS_KERNEL_ENTRY,
                                           FLOATING_DATA_TYPE_SEQ,
                                           INT_DATA_TYPE_SEQ)};
  return data_type2creator.at(
      GetHashKey(op_conf.softmax_loss_conf().prediction().data_type(),
                 op_conf.softmax_loss_conf().label().data_type()))();
}

}  // namespace

REGISTER_TEMPLATE_KERNEL_CREATOR(OperatorConf::kSoftmaxLossConf,
                                 CreateSoftmaxLossKernel);

}  // namespace oneflow
