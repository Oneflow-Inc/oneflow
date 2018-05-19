#include "oneflow/core/kernel/reduce_add_kernel.h"

namespace {

bool ExistingFirstRegstInPiece(int64_t processed_regsts_cnt, int64_t regsts_num_per_piece) {
  return processed_regsts_cnt % regsts_num_per_piece == 0;
}

}  // namespace

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceAddKernel<device_type, T>::VirtualKernelInit(const ParallelContext* ctx) {
  parallel_id_ = ctx->parallel_id();
}

template<DeviceType device_type, typename T>
void ReduceAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& input_bns = this->op_attribute().input_bns();
  int64_t processed_regsts_cnt = reinterpret_cast<int64_t>(ctx.other);
  bool is_first_add = ExistingFirstRegstInPiece(processed_regsts_cnt, input_bns.size());

  Blob* copy_buf_blob = BnInOp2Blob(this->op_attribute().data_tmp_bns().Get(0));
  Blob* out_blob = BnInOp2Blob("out");
  int64_t elem_cnt = out_blob->shape().elem_cnt();
  FOR_RANGE(int32_t, i, 0, input_bns.size()) {
    Blob* in_blob = BnInOp2Blob(input_bns.Get(i));
    if (in_blob == nullptr) { continue; }
    if (is_first_add) {
      AutoMemcpy(ctx.device_ctx, out_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                 in_blob->ByteSizeOfDataContentField(), in_blob->mem_case(), out_blob->mem_case());
      is_first_add = false;
      continue;
    }
    if (i == parallel_id_) {
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, elem_cnt, 1.0, in_blob->dptr<T>(), 1,
                                       out_blob->mut_dptr<T>(), 1);

    } else {
      AutoMemcpy(ctx.device_ctx, copy_buf_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                 in_blob->ByteSizeOfDataContentField(), in_blob->mem_case(),
                 copy_buf_blob->mem_case());
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, elem_cnt, 1.0, copy_buf_blob->dptr<T>(), 1,
                                       out_blob->mut_dptr<T>(), 1);
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceAddConf, ReduceAddKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
