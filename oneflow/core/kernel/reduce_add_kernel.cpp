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
  int64_t processed_regst_cnt = reinterpret_cast<int64_t>(ctx.other);
  int64_t piece_id = processed_regst_cnt / input_bns.size();
  bool is_first_add = processed_regst_cnt % input_bns.size() == 0;

  Blob* copy_buf_blob = BnInOp2Blob(this->op_attribute().data_tmp_bns().Get(0));
  Blob* out_blob = BnInOp2Blob("out");
  int64_t elem_cnt = out_blob->shape().elem_cnt();
  FOR_RANGE(int32_t, i, 0, input_bns.size()) {
    Blob* in_blob = BnInOp2Blob(input_bns.Get(i));
    if (in_blob == nullptr || in_blob->piece_id() != piece_id) { continue; }
    bool need_copy_h2d =
        in_blob->mem_case().has_host_mem() && out_blob->mem_case().has_device_cuda_mem();
    if (is_first_add) {
      if (need_copy_h2d) {
        Memcpy<DeviceType::kGPU>(ctx.device_ctx, out_blob->mut_dptr<T>(), in_blob->dptr<T>(),
                                 in_blob->ByteSizeOfDataContentField(),
                                 cudaMemcpyKind::cudaMemcpyHostToDevice);
      } else {
        Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                            out_blob->ByteSizeOfDataContentField());
      }
      is_first_add = false;
      continue;
    }
    if (i == parallel_id_) {
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, elem_cnt, 1.0, in_blob->dptr<T>(), 1,
                                       out_blob->mut_dptr<T>(), 1);

    } else {
      Memcpy<DeviceType::kGPU>(ctx.device_ctx, copy_buf_blob->mut_dptr<T>(), in_blob->dptr<T>(),
                               in_blob->ByteSizeOfDataContentField(),
                               cudaMemcpyKind::cudaMemcpyHostToDevice);
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, elem_cnt, 1.0, copy_buf_blob->dptr<T>(), 1,
                                       out_blob->mut_dptr<T>(), 1);
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceAddConf, ReduceAddKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
