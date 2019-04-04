#include "oneflow/core/kernel/keep_header_only_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void KeepHeaderOnlyKernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  this->CopyField(ctx.device_ctx, BnInOp2Blob, this->op_attribute().input_bns(),
            this->op_attribute().output_bns(), &Blob::CopyDim0ValidNumFrom);
}

template<DeviceType device_type>
void KeepHeaderOnlyKernel<device_type>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  this->CopyField(ctx.device_ctx, BnInOp2Blob, this->op_attribute().input_bns(),
            this->op_attribute().output_bns(), &Blob::CopyDim1ValidNumFrom);
}

template<DeviceType device_type>
void KeepHeaderOnlyKernel<device_type>::ForwardDim2ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  this->CopyField(ctx.device_ctx, BnInOp2Blob, this->op_attribute().input_bns(),
            this->op_attribute().output_bns(), &Blob::CopyDim2ValidNumFrom);
}

template<DeviceType device_type>
void KeepHeaderOnlyKernel<device_type>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  this->CopyField(ctx.device_ctx, BnInOp2Blob, this->op_attribute().input_bns(),
            this->op_attribute().output_bns(), &Blob::CopyRecordIdInDevicePieceFrom);
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kKeepHeaderOnlyConf, KeepHeaderOnlyKernel);

}
