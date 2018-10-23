#include "oneflow/core/kernel/batch_permutation_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BatchPermutation<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* indices_blob = BnInOp2Blob("indices");
  Blob* out_blob = BnInOp2Blob("out");
  BatchPermutationUtil<device_type, T>::Forward(ctx, this->op_conf().batch_permutation_conf(),
                                                in_blob, indices_blob, out_blob);
}

template<DeviceType device_type, typename T>
void BatchPermutation<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* indices_blob = BnInOp2Blob("indices");
  BatchPermutationUtil<device_type, T>::Backward(ctx, this->op_conf().batch_permutation_conf(),
                                                 out_diff_blob, indices_blob, in_diff_blob);
}

template<typename T>
struct BatchPermutationUtil<DeviceType::kCPU, T> {
  static void Forward(const KernelCtx& ctx, const BatchPermutationOpConf& conf, const Blob* in_blob,
                      const Blob* indices_blob, Blob* out_blob) {
    int32_t channel_area = in_blob->shape().At(1) * in_blob->shape().At(2) * in_blob->shape().At(3);
    size_t channel_size = channel_area * sizeof(T);
    for (int i = 0; i < in_blob->shape().At(0); i++) {
      int32_t idx = indices_blob->dptr<int32_t>()[i];
      Memcpy<DeviceType::kCPU>(ctx.device_ctx, out_blob->mut_dptr<T>() + i * channel_area,
                               in_blob->dptr<T>() + idx * channel_area, channel_size);
    }
  }

  static void Backward(const KernelCtx& ctx, const BatchPermutationOpConf& conf,
                       const Blob* out_diff_blob, const Blob* indices_blob, Blob* in_diff_blob) {
    int32_t channel_area =
        out_diff_blob->shape().At(1) * out_diff_blob->shape().At(2) * out_diff_blob->shape().At(3);
    size_t channel_size = channel_area * sizeof(T);
    for (int i = 0; i < out_diff_blob->shape().At(0); i++) {
      int32_t idx = indices_blob->dptr<int32_t>()[i];
      Memcpy<DeviceType::kCPU>(ctx.device_ctx, in_diff_blob->mut_dptr<T>() + idx * channel_area,
                               out_diff_blob->dptr<T>() + i * channel_area, channel_size);
    }
  }
};

template<DeviceType device_type, typename T>
void BatchPermutation<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK_EQ(BnInOp2Blob("in")->dim0_valid_num(0), BnInOp2Blob("indices")->dim0_valid_num(0));
  BnInOp2Blob("out")->set_dim0_valid_num(0, BnInOp2Blob("indices")->dim0_valid_num(0));
}

template<DeviceType device_type, typename T>
void BatchPermutation<device_type, T>::BackwardInDiffDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob(GenDiffBn("in"))->set_dim0_valid_num(0, BnInOp2Blob("indices")->dim0_valid_num(0));
}

template<DeviceType device_type, typename T>
void BatchPermutation<device_type, T>::ForwardRecordIdxInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyRecordIdxInDevicePieceFrom(ctx.device_ctx, BnInOp2Blob("indices"));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBatchPermutationConf, BatchPermutation,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
