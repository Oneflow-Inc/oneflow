#include "oneflow/core/kernel/extract_piece_slice_id_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ExtractPieceSliceIdKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->op_conf().extract_piece_slice_id_conf();
  FOR_RANGE(int32_t, i, 0, conf.in_size()) {
    const Blob* in_i_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(i));
    Blob* out_i_blob = BnInOp2Blob(this->op_attribute().output_bns().Get(i));
    ExtractPieceSliceIdUtil<device_type>::ForwardOneInOutPair(
        ctx.device_ctx, in_i_blob->shape().At(0), i, out_i_blob->mut_dptr<int32_t>());
  }
}

template<DeviceType device_type>
void ExtractPieceSliceIdKernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->op_conf().extract_piece_slice_id_conf();
  FOR_RANGE(int32_t, i, 0, conf.in_size()) {
    BnInOp2Blob(this->op_attribute().output_bns().Get(i))
        ->CopyDim0ValidNumFrom(ctx.device_ctx,
                               BnInOp2Blob(this->op_attribute().input_bns().Get(i)));
  }
}

void ExtractPieceSliceIdUtil<DeviceType::kCPU>::ForwardOneInOutPair(DeviceCtx* ctx,
                                                                    const int32_t instance_num,
                                                                    const int32_t slice_idx,
                                                                    int32_t* out_i_ptr) {
  FOR_RANGE(int32_t, i, 0, instance_num) { out_i_ptr[i] = slice_idx; }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kExtractPieceSliceIdConf, ExtractPieceSliceIdKernel);

}  // namespace oneflow
