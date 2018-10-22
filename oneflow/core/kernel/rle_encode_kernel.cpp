#include "oneflow/core/kernel/rle_encode_kernel.h"
extern "C" {
#include <maskApi.h>
}

namespace oneflow {

void RleEncodeKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  CHECK_EQ(in->shape().NumAxes(), 3);
  const auto height = static_cast<size_t>(in->shape().At(1));
  const auto width = static_cast<size_t>(in->shape().At(2));
  FOR_RANGE(int64_t, i, 0, in->shape().At(0)) {
    RLE rle;
    rleEncode(&rle, in->dptr<uint8_t>(i), height, width, 1);
    char* str = rleToString(&rle);
    const size_t len = strlen(str);
    CHECK_LE(len, out->shape().At(1));
    Memcpy<DeviceType::kCPU>(ctx.device_ctx, out->mut_dptr<char>(i), str, len);
    out->set_dim1_valid_num(i, len);
    free(str);
    rleFree(&rle);
  }
};

void RleEncodeKernel::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  out->set_dim0_valid_num(0, in->dim0_valid_num(0));
}

void RleEncodeKernel::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

REGISTER_KERNEL(OperatorConf::kRleEncodeConf, RleEncodeKernel);

}  // namespace oneflow
