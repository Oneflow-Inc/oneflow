#include "oneflow/core/kernel/embedding_lookup_mddiffacc_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void EmbeddingLookupMdDiffAccKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* one_ids_blob = BnInOp2Blob("one_ids");
  const Blob* one_val_blob = BnInOp2Blob("one_val");
  Blob* acc_ids_blob = BnInOp2Blob("acc_ids");
  Blob* acc_val_blob = BnInOp2Blob("acc_val");
  int32_t one_ids_size = one_ids_blob->shape().elem_cnt();
  int32_t one_val_size = one_val_blob->shape().elem_cnt();
  int32_t offset = *reinterpret_cast<int32_t*>(ctx.other);
  Memcpy<device_type>(ctx.device_ctx, acc_ids_blob->mut_dptr<T>() + offset * one_ids_size,
                      one_ids_blob->dptr<T>(), sizeof(T) * one_ids_size);
  Memcpy<device_type>(ctx.device_ctx, acc_val_blob->mut_dptr<T>() + offset * one_val_size,
                      one_val_blob->dptr<T>(), sizeof(T) * one_val_size);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kEmbeddingLookupAccumulateConf,
                           EmbeddingLookupMdDiffAccKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
