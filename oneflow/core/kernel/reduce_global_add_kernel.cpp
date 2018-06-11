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
  Blob* out_blob = BnInOp2Blob("out");
  Blob* middle_blob = BnInOp2Blob("middle");
  if (middle_blob) {
    Memset<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(), 0,
                        out_blob->ByteSizeOfDataContentField());
  } else {
    middle_blob = BnInOp2Blob("in_" + std::to_string(parallel_id_));
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(), middle_blob->dptr<char>(),
                        out_blob->ByteSizeOfDataContentField());
  }
  int64_t elem_cnt = out_blob->shape().elem_cnt();
  for (const std::string& input_bn : this->op_attribute().input_bns()) {
    if (input_bn == "in_" + std::to_string(parallel_id_)) { continue; }
    Blob* in_blob = BnInOp2Blob(input_bn);
    AutoMemcpy(ctx.device_ctx, middle_blob->mut_dptr<char>(), in_blob->mut_dptr<char>(),
               in_blob->ByteSizeOfDataContentField(), in_blob->mem_case(), middle_blob->mem_case());
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, elem_cnt, 1.0, middle_blob->dptr<T>(), 1,
                                     out_blob->mut_dptr<T>(), 1);
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceGlobalAddConf, ReduceGlobalAddKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
