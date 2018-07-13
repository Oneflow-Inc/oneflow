#include "oneflow/core/kernel/reduce_global_add_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceGlobalAddKernel<device_type, T>::VirtualKernelInit(const ParallelContext* parallel_ctx,
                                                              DeviceCtx* device_ctx) {
  parallel_id_ = parallel_ctx->parallel_id();
}

template<DeviceType device_type, typename T>
void ReduceGlobalAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto* other_val = static_cast<std::pair<std::string, bool>*>(ctx.other);
  std::string input_bn = other_val->first;
  bool is_first = other_val->second;

  Blob* out_blob = BnInOp2Blob("out");
  Blob* in_blob = BnInOp2Blob(input_bn);
  if (is_first) {
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                        out_blob->ByteSizeOfDataContentField());
  } else {
    int64_t elem_cnt = out_blob->shape().elem_cnt();
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, elem_cnt, 1.0, in_blob->dptr<T>(), 1,
                                     out_blob->mut_dptr<T>(), 1);
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceGlobalAddConf, ReduceGlobalAddKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
