#include "oneflow/core/kernel/binary_cross_entropy_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
void Forward(DeviceCtx* ctx, const Blob* prediction, const Blob* label, Blob* out) {
  const int64_t num_instances = label->shape().elem_cnt();
  CHECK_EQ(prediction->shape().elem_cnt(), num_instances);
  BinaryCrossEntropyKernelUtil<device_type, T, K>::ComputeEntropy(
      ctx, num_instances, prediction->dptr<T>(), label->dptr<K>(), out->mut_dptr<T>());
}

template<DeviceType device_type, typename T, typename K>
void Backward(DeviceCtx* ctx, const Blob* prediction, const Blob* label, const Blob* out_diff,
              Blob* prediction_diff) {
  const int64_t num_instances = label->shape().elem_cnt();
  CHECK_EQ(prediction->shape().elem_cnt(), num_instances);
  BinaryCrossEntropyKernelUtil<device_type, T, K>::ComputeDiff(
      ctx, num_instances, prediction->dptr<T>(), label->dptr<K>(), out_diff->dptr<T>(),
      prediction_diff->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
struct BinaryCrossEntropyUntil final {
#define MAKE_CROSS_ENTROPY_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
  DEFINE_STATIC_SWITCH_FUNC(void, Forward, MAKE_CROSS_ENTROPY_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
  DEFINE_STATIC_SWITCH_FUNC(void, Backward, MAKE_CROSS_ENTROPY_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
#undef MAKE_CROSS_ENTROPY_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
void BinaryCrossEntropyKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  Blob* out = BnInOp2Blob("out");
  BinaryCrossEntropyUntil<device_type, T>::SwitchForward(SwitchCase(label->data_type()),
                                                         ctx.device_ctx, prediction, label, out);
}

template<DeviceType device_type, typename T>
void BinaryCrossEntropyKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  const Blob* out_diff = BnInOp2Blob(GenDiffBn("out"));
  Blob* prediction_diff = BnInOp2Blob(GenDiffBn("prediction"));
  BinaryCrossEntropyUntil<device_type, T>::SwitchBackward(
      SwitchCase(label->data_type()), ctx.device_ctx, prediction, label, out_diff, prediction_diff);
}

template<typename T, typename K>
struct BinaryCrossEntropyKernelUtil<DeviceType::kCPU, T, K> {
  static void ComputeEntropy(DeviceCtx* ctx, int64_t num_instances, const T* x, const K* labels,
                             T* y) {
    FOR_RANGE(int64_t, i, 0, num_instances) {
      const K label = labels[i];
      CHECK(label == 0 || label == 1);
      y[i] = -SafeLog(label == 0 ? OneVal<T>::value - x[i] : x[i]);
    }
  }

  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, const T* x, const K* labels,
                          const T* dy, T* dx) {
    FOR_RANGE(int64_t, i, 0, num_instances) {
      const K label = labels[i];
      CHECK(label == 0 || label == 1);
      dx[i] = -dy[i] / MaxWithLogThreshold(label == 0 ? OneVal<T>::value - x[i] : x[i]);
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBinaryCrossEntropyConf, BinaryCrossEntropyKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
