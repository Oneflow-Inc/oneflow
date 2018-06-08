#include "oneflow/core/kernel/reduce_local_add_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceLocalAddKernel<device_type, T>::VirtualKernelInit(const ParallelContext* ctx) {
  parallel_id_ = ctx->parallel_id();
}

template<DeviceType device_type, typename T>
void ReduceLocalAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& input_bns = this->op_attribute().input_bns();
  int64_t min_in_parallel_id = this->op_conf().reduce_local_add_conf().min_in_parallel_id();
  Blob* same_parallel_in_blob = BnInOp2Blob(input_bns.Get(parallel_id_ - min_in_parallel_id));
  Blob* middle_blob = BnInOp2Blob("middle");
  Memcpy<device_type>(ctx.device_ctx, middle_blob->mut_dptr<char>(),
                      same_parallel_in_blob->dptr<char>(),
                      middle_blob->ByteSizeOfDataContentField());
  int64_t elem_cnt = same_parallel_in_blob->shape().elem_cnt();
  FOR_RANGE(int32_t, i, 0, input_bns.size()) {
    if (i + min_in_parallel_id == parallel_id_) { continue; }
    Blob* in_blob = BnInOp2Blob(input_bns.Get(i));
    Blob* src_blob = in_blob;
    if (in_blob->mem_case().has_host_mem() && middle_blob->mem_case().has_device_cuda_mem()) {
      Memcpy<DeviceType::kGPU>(ctx.device_ctx, same_parallel_in_blob->mut_dptr<T>(),
                               in_blob->dptr<T>(), in_blob->ByteSizeOfDataContentField(),
                               cudaMemcpyKind::cudaMemcpyHostToDevice);
      src_blob = same_parallel_in_blob;
    }
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, elem_cnt, 1.0, src_blob->dptr<T>(), 1,
                                     middle_blob->mut_dptr<T>(), 1);
  }
  const PbRpf<std::string>& output_bns = this->op_attribute().output_bns();
  const char* src_middle_dptr = middle_blob->dptr<char>();
  FOR_RANGE(int32_t, i, 0, output_bns.size()) {
    Blob* out_blob = BnInOp2Blob(output_bns.Get(i));
    size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
    AutoMemcpy(ctx.device_ctx, out_blob->mut_dptr<char>(), src_middle_dptr, out_byte_size,
               middle_blob->mem_case(), out_blob->mem_case());
    src_middle_dptr += out_byte_size;
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceLocalAddConf, ReduceLocalAddKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
