#include "oneflow/core/kernel/sparse_softmax_cross_entropy_kernel.h"
#include "oneflow/core/kernel/sparse_cross_entropy_kernel_util.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
void Forward(DeviceCtx* ctx, const Blob* prediction, const Blob* label, Blob* tmp, Blob* buf,
             Blob* prob, Blob* out) {
  const int64_t n = prediction->shape().At(0);
  const int64_t w = prediction->shape().Count(1);
  Memset<device_type>(ctx, out->mut_dptr<T>(), 0, out->ByteSizeOfDataContentField());
  SoftmaxComputeProb<device_type, T>(ctx, n, w, prediction->dptr<T>(), tmp->mut_dptr<T>(),
                                     prob->mut_dptr<T>(), buf->mut_dptr(),
                                     buf->ByteSizeOfDataContentField());
  SparseCrossEntropyKernelUtil<device_type, T, K>::ComputeEntropy(
      ctx, n, w, prob->dptr<T>(), label->dptr<K>(), out->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
struct SparseSoftmaxCrossEntropyUntil final {
#define MAKE_SOFTMAX_CROSS_ENTROPY_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
  DEFINE_STATIC_SWITCH_FUNC(void, Forward, MAKE_SOFTMAX_CROSS_ENTROPY_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
#undef MAKE_SOFTMAX_CROSS_ENTROPY_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
void SparseSoftmaxCrossEntropyKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction_blob = BnInOp2Blob("prediction");
  const Blob* label_blob = BnInOp2Blob("label");
  Blob* tmp_blob = BnInOp2Blob("fw_softmax_num");
  Blob* buf_blob = BnInOp2Blob("fw_buf");
  Blob* prob_blob = BnInOp2Blob("prob");
  Blob* out_blob = BnInOp2Blob("out");
  SparseSoftmaxCrossEntropyUntil<device_type, T>::SwitchForward(
      SwitchCase(label_blob->data_type()), ctx.device_ctx, prediction_blob, label_blob, tmp_blob,
      buf_blob, prob_blob, out_blob);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSparseSoftmaxCrossEntropyConf,
                           SparseSoftmaxCrossEntropyKernel, FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSparseSoftmaxCrossEntropyMs1Conf,
                           SparseSoftmaxCrossEntropyKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
