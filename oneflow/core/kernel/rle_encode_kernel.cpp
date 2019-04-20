#include "oneflow/core/kernel/rle_encode_kernel.h"
#include "oneflow/core/kernel/rle_util.h"

namespace oneflow {

void RleEncodeKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  const Blob* size = BnInOp2Blob("size");
  Blob* out = BnInOp2Blob("out");

  const size_t max_len = static_cast<size_t>(out->static_shape().At(1));
  const uint8_t* in_ptr = in->dptr<uint8_t>();
  FOR_RANGE(int32_t, i, 0, out->shape().At(0)) {
    const int32_t im_idx = in->record_id_in_device_piece(i);
    const int32_t height = size->dptr<int32_t>(im_idx)[0];
    const int32_t width = size->dptr<int32_t>(im_idx)[1];
    const int32_t hxw = in->static_shape().Count(1);
    CHECK_LE(height * width, hxw);
    const size_t len =
        RleUtil::EncodeToString(in_ptr + i * hxw, height, width, max_len, out->mut_dptr<char>(i));
    out->set_dim1_valid_num(i, len);
  }
};

void RleEncodeKernel::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->set_dim0_valid_num(0, BnInOp2Blob("in")->dim0_valid_num(0));
}

void RleEncodeKernel::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

void RleEncodeKernel::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyRecordIdInDevicePieceFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

REGISTER_KERNEL(OperatorConf::kRleEncodeConf, RleEncodeKernel);

}  // namespace oneflow
