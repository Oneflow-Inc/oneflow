#include "oneflow/core/kernel/rle_encode_kernel.h"
#include "oneflow/core/kernel/rle_util.h"

namespace oneflow {

void RleEncodeKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  CHECK_EQ(in->shape().NumAxes(), 3);
  const size_t height = static_cast<size_t>(in->shape().At(1));
  const size_t width = static_cast<size_t>(in->shape().At(2));
  CHECK_EQ(out->shape().NumAxes(), 2);
  const size_t max_len = static_cast<size_t>(out->shape().At(1));
  FOR_RANGE(int64_t, i, 0, in->shape().At(0)) {
    const size_t len = RleUtil::EncodeToString(in->dptr<uint8_t>(i), height, width, max_len,
                                               out->mut_dptr<char>(i));
    out->set_dim1_valid_num(i, len);
  }
};

void RleEncodeKernel::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

void RleEncodeKernel::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

REGISTER_KERNEL(OperatorConf::kRleEncodeConf, RleEncodeKernel);

}  // namespace oneflow
