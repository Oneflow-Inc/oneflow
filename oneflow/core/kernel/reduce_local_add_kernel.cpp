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

  const PbRpf<std::string>& output_bns = this->op_attribute().output_bns();
  size_t byte_offset = 0;
  int64_t elem_offset = 0;
  FOR_RANGE(int32_t, i, 0, output_bns.size()) {
    Blob* out_blob = BnInOp2Blob(output_bns.Get(i));
    size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
    int64_t out_elem_cnt = out_blob->shape().elem_cnt();

    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(),
                        same_parallel_in_blob->dptr<char>() + byte_offset, out_byte_size);
    FOR_RANGE(int32_t, j, 0, input_bns.size()) {
      if (j + min_in_parallel_id == parallel_id_) { continue; }
      Blob* src_blob = BnInOp2Blob(input_bns.Get(j));
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, out_elem_cnt, 1.0,
                                       src_blob->dptr<T>() + elem_offset, 1,
                                       out_blob->mut_dptr<T>(), 1);
    }

    byte_offset += out_byte_size;
    elem_offset += out_elem_cnt;
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceLocalAddConf, ReduceLocalAddKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
