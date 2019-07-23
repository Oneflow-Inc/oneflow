#include "oneflow/core/kernel/sparse_cross_entropy_kernel.h"
#include "oneflow/core/kernel/sparse_cross_entropy_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
void Forward(DeviceCtx* ctx, const Blob* prediction, const Blob* label, Blob* out) {
  const int64_t num_instances = label->shape().elem_cnt();
  CHECK_EQ(prediction->shape().elem_cnt() % num_instances, 0);
  const int64_t num_classes = prediction->shape().elem_cnt() / num_instances;
  SparseCrossEntropyKernelUtil<device_type, T, K>::ComputeEntropy(
      ctx, num_instances, num_classes, prediction->dptr<T>(), label->dptr<K>(), out->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
struct SparseCrossEntropyUntil final {
#define MAKE_CROSS_ENTROPY_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
  DEFINE_STATIC_SWITCH_FUNC(void, Forward, MAKE_CROSS_ENTROPY_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
#undef MAKE_CROSS_ENTROPY_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
void SparseCrossEntropyKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  Blob* out = BnInOp2Blob("out");
  SparseCrossEntropyUntil<device_type, T>::SwitchForward(SwitchCase(label->data_type()),
                                                         ctx.device_ctx, prediction, label, out);
}

ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(OperatorConf::kSparseCrossEntropyConf,
                                         SparseCrossEntropyKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
