#include "oneflow/core/operator/reduce_split_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

// TODO: put gcd() and lcm() to some xxx_util file
namespace {

const int64_t gcd(int64_t a, int64_t b) {
  while (true) {
    if (a == 0) { return b; }
    b %= a;
    if (b == 0) { return a; }
    a %= b;
  }
}

const int64_t lcm(int64_t a, int64_t b) {
  const int64_t tmp = gcd(a, b);
  return tmp ? (a / tmp * b) : 0;
}

}  // namespace

void ReduceSplitOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_split_conf());
  for (int32_t i = 0; i < op_conf().reduce_split_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
  EnrollInputBn("in", false);
}

const PbMessage& ReduceSplitOp::GetCustomizedConf() const { return op_conf().reduce_split_conf(); }

void ReduceSplitOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  ReduceSplitKernelConf* reduce_split_conf = kernel_conf->mutable_reduce_split_conf();
  int64_t offset = 0;
  for (int32_t i = 0; i < op_conf().reduce_split_conf().out_num(); ++i) {
    reduce_split_conf->mutable_data_offset()->Add(offset);
    offset += RtBlobDesc(*(GetBlobDesc4BnInOp(output_bns().Get(i)))).ByteSizeOfBlobBody();
  }
  // we don't add data_type_byte_size to round_up_block_size because we assume
  // kCudaAlignSize % data_type_byte_size == 0
  const int64_t round_up_block_size = lcm(parallel_ctx->parallel_num(), kCudaAlignSize);
  const int64_t in_blob_body_size = RoundUp(offset, round_up_block_size);
  const int64_t data_type_byte_size =
      static_cast<int64_t>(GetSizeOfDataType(GetBlobDesc4BnInOp(SoleIbn())->data_type()));
  CHECK_EQ(in_blob_body_size % data_type_byte_size, 0);
  const int64_t in_blob_elem_cnt = in_blob_body_size / data_type_byte_size;
  CHECK_EQ(in_blob_elem_cnt * data_type_byte_size,
           RtBlobDesc(*GetBlobDesc4BnInOp(SoleIbn())).ByteSizeOfBlobBody());
}

REGISTER_OP(OperatorConf::kReduceSplitConf, ReduceSplitOp);

}  // namespace oneflow
