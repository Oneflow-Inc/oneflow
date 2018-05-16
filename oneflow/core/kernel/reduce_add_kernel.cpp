#include "oneflow/core/kernel/reduce_add_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceAddKernel<device_type, T>::VirtualKernelInit(const ParallelContext* ctx) {
  parallel_id_ = ctx->parallel_id();
}

template<DeviceType device_type, typename T>
void ReduceAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& input_bns = this->op_attribute().input_bns();
  Blob* same_parallel_in_blob = BnInOp2Blob(input_bns.Get(parallel_id_));
  Blob* out_blob = BnInOp2Blob("out");
  Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(),
                      same_parallel_in_blob->dptr<char>(), out_blob->ByteSizeOfDataContentField());
  int64_t elem_cnt = out_blob->shape().elem_cnt();
  for (int32_t i = 0; i < input_bns.size(); ++i) {
    if (i == parallel_id_) { continue; }
    Blob* in_blob = BnInOp2Blob(input_bns.Get(i));
    Blob* src_blob = in_blob;
    if (in_blob->mem_case().has_host_mem() && out_blob->mem_case().has_device_cuda_mem()) {
      Memcpy<DeviceType::kGPU>(ctx.device_ctx, same_parallel_in_blob->mut_dptr<T>(),
                               in_blob->dptr<T>(), in_blob->ByteSizeOfDataContentField(),
                               cudaMemcpyKind::cudaMemcpyHostToDevice);
      src_blob = same_parallel_in_blob;
    }
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, elem_cnt, 1.0, src_blob->dptr<T>(), 1,
                                     out_blob->mut_dptr<T>(), 1);
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceAddConf, ReduceAddKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
