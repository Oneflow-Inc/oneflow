#include "oneflow/core/kernel/sparse_cross_entropy_kernel_util.h"
#include "oneflow/core/kernel/sparse_cross_entropy_kernel.h"
#include "oneflow/core/kernel/sparse_cross_entropy_grad_kernel.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
void Backward(DeviceCtx* ctx, const int64_t lower_bound, const Blob* prediction, const Blob* label,
              const Blob* dy, Blob* prediction_diff) {
  const int64_t num_instances = label->shape().elem_cnt();
  CHECK_EQ(prediction->shape().elem_cnt() % num_instances, 0);
  const int64_t num_classes = prediction->shape().elem_cnt() / num_instances;
  Memset<device_type>(ctx, prediction_diff->mut_dptr<T>(), 0,
                      prediction_diff->ByteSizeOfDataContentField());
  SparseCrossEntropyKernelUtil<device_type, T, K>::ComputeDiff(
      ctx, num_instances, num_classes, prediction->dptr<T>(), label->dptr<K>(), dy->dptr<T>(),
      prediction_diff->mut_dptr<T>(), lower_bound);
}

template<DeviceType device_type, typename T>
struct SparseCrossEntropyUntil final {
#define MAKE_CROSS_ENTROPY_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
  DEFINE_STATIC_SWITCH_FUNC(void, Backward, MAKE_CROSS_ENTROPY_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
#undef MAKE_CROSS_ENTROPY_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
void SparseCrossEntropyGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  const Blob* dy = BnInOp2Blob("dy");
  Blob* prediction_diff = BnInOp2Blob("prediction_diff");
  int64_t lower_bound = 0;
  if (this->kernel_conf().has_sparse_cross_entropy_grad_conf()) {
    lower_bound = this->kernel_conf().sparse_cross_entropy_grad_conf().lower_bound();
  }
  SparseCrossEntropyUntil<device_type, T>::SwitchBackward(SwitchCase(label->data_type()),
                                                          ctx.device_ctx, lower_bound, prediction,
                                                          label, dy, prediction_diff);
}

ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(OperatorConf::kSparseCrossEntropyGradConf,
                                         SparseCrossEntropyGradKernel, FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(OperatorConf::kSparseCrossEntropyMs1GradConf,
                                         SparseCrossEntropyGradKernel, FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
