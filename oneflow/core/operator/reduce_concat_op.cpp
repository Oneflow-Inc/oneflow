#include "oneflow/core/operator/reduce_concat_op.h"
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

void ReduceConcatOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_concat_conf());
  for (int32_t i = 0; i < op_conf().reduce_concat_conf().in_num(); ++i) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  EnrollOutputBn("out", false);
}

const PbMessage& ReduceConcatOp::GetCustomizedConf() const {
  return op_conf().reduce_concat_conf();
}

void ReduceConcatOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx,
                                    std::function<void(OpContext*)> EnrollOpCtx) const {
  const BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  const DataType data_type = first_in_blob->data_type();
  for (int32_t i = 1; i < op_conf().reduce_concat_conf().in_num(); ++i) {
    CHECK_EQ(data_type, GetBlobDesc4BnInOp(input_bns().Get(i))->data_type());
  }

  BlobDesc* out_blob = GetBlobDesc4BnInOp(SoleObn());
  *out_blob = *first_in_blob;
  int64_t in_blob_body_size_sum = 0;
  for (int32_t i = 0; i < op_conf().reduce_concat_conf().in_num(); ++i) {
    in_blob_body_size_sum +=
        RtBlobDesc(*(GetBlobDesc4BnInOp(input_bns().Get(i)))).ByteSizeOfBlobBody();
  }
  // we don't add data_type_byte_size to round_up_block_size because we assume
  // kCudaAlignSize % data_type_byte_size == 0
  const int64_t round_up_block_size = lcm(parallel_ctx->parallel_num(), kCudaAlignSize);
  const int64_t out_blob_body_size = RoundUp(in_blob_body_size_sum, round_up_block_size);
  const int64_t data_type_byte_size =
      static_cast<int64_t>(GetSizeOfDataType(first_in_blob->data_type()));
  CHECK_EQ(out_blob_body_size % data_type_byte_size, 0);
  const int64_t out_blob_elem_cnt = out_blob_body_size / data_type_byte_size;
  out_blob->mut_shape() = Shape({out_blob_elem_cnt});

  ReduceConcatOpCtx* reduce_concat_op_ctx =
      new ReduceConcatOpCtx(data_type_byte_size, out_blob_elem_cnt);
  EnrollOpCtx(reduce_concat_op_ctx);
}

void ReduceConcatOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  ReduceConcatKernelConf* reduce_concat_conf = kernel_conf->mutable_reduce_concat_conf();
  int64_t offset = 0;
  for (int32_t i = 0; i < op_conf().reduce_concat_conf().in_num(); ++i) {
    reduce_concat_conf->mutable_data_offset()->Add(offset);
    offset += RtBlobDesc(*(GetBlobDesc4BnInOp(input_bns().Get(i)))).ByteSizeOfBlobBody();
  }
  const ReduceConcatOpCtx* reduce_concat_op_ctx = static_cast<const ReduceConcatOpCtx*>(op_ctx);
  CHECK_EQ((reduce_concat_op_ctx->data_type_byte_size) * (reduce_concat_op_ctx->out_blob_elem_cnt),
           RtBlobDesc(*GetBlobDesc4BnInOp(SoleObn())).ByteSizeOfBlobBody());
}

LogicalBlobId ReduceConcatOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name("out");
  return ret;
}

REGISTER_OP(OperatorConf::kReduceConcatConf, ReduceConcatOp);

}  // namespace oneflow
