#include "oneflow/core/kernel/sparse_softmax_cross_entropy_grad_kernel.h"
#include "oneflow/core/kernel/sparse_cross_entropy_kernel.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
void Backward(DeviceCtx* ctx, const int64_t lower_bound, const Blob* dy, const Blob* label,
              const Blob* prob, Blob* dx) {
  const int64_t n = dx->shape().At(0);
  const int64_t w = dx->shape().Count(1);
  T* dx_dptr = dx->mut_dptr<T>();
  KernelUtil<device_type, T>::Copy(ctx, n * w, prob->dptr<T>(), 1, dx_dptr, 1);
  SparseSoftmaxCrossEntropyGradKernelUtil<device_type, T, int32_t>::BackwardSub(
      ctx, n, w, lower_bound, dy->dptr<T>(), label->dptr<int32_t>(), dx_dptr);
}

template<DeviceType device_type, typename T>
struct SparseSoftmaxCrossEntropyUntil final {
#define MAKE_SOFTMAX_CROSS_ENTROPY_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
  DEFINE_STATIC_SWITCH_FUNC(void, Backward, MAKE_SOFTMAX_CROSS_ENTROPY_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
#undef MAKE_SOFTMAX_CROSS_ENTROPY_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
void SparseSoftmaxCrossEntropyGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* dy_blob = BnInOp2Blob("dy");
  const Blob* label_blob = BnInOp2Blob("label");
  const Blob* prob_blob = BnInOp2Blob("prob");
  Blob* dx_blob = BnInOp2Blob("dx");
  int64_t lower_bound = 0;
  if (this->kernel_conf().has_sparse_softmax_cross_entropy_grad_conf()) {
    lower_bound = this->kernel_conf().sparse_softmax_cross_entropy_grad_conf().lower_bound();
  }
  SparseSoftmaxCrossEntropyUntil<device_type, T>::SwitchBackward(
      SwitchCase(label_blob->data_type()), ctx.device_ctx, lower_bound, dy_blob, label_blob,
      prob_blob, dx_blob);
}

template<typename T, typename K>
struct SparseSoftmaxCrossEntropyGradKernelUtil<DeviceType::kCPU, T, K> {
  static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const int64_t lower_bound, const T* dy, const K* label, T* in_diff) {
    for (int64_t i = 0; i < n; ++i) {
      const int64_t idx = label[i] - lower_bound;
      if (idx >= 0 && idx < w) { in_diff[i * w + idx] = dy[i] * (in_diff[i * w + idx] - 1); }
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSparseSoftmaxCrossEntropyGradConf,
                           SparseSoftmaxCrossEntropyGradKernel, FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSparseSoftmaxCrossEntropyMs1GradConf,
                           SparseSoftmaxCrossEntropyGradKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
