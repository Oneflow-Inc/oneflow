#include "oneflow/core/operator/reduce_concat_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

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
                                    const ParallelContext* parallel_ctx) const {
  int32_t in_num = op_conf().reduce_concat_conf().in_num();
  BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  BlobDesc* out_blob = GetBlobDesc4BnInOp(SoleObn());
  *out_blob = *first_in_blob;
  int64_t out_blob_elem_cnt = first_in_blob->shape().elem_cnt();
  for (int32_t i = 1; i < in_num; ++i) {
    out_blob_elem_cnt += GetBlobDesc4BnInOp(input_bns().Get(i))->shape().elem_cnt();
  }
  out_blob->mut_shape() = Shape({out_blob_elem_cnt});
}

void ReduceConcatOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf) const {
  ReduceConcatKernelConf* reduce_concat_conf = kernel_conf->mutable_reduce_concat_conf();
  int64_t offset = 0;
  for (int32_t i = 0; i < op_conf().reduce_concat_conf().in_num(); ++i) {
    reduce_concat_conf->mutable_data_offset()->Add(offset);
    offset += RtBlobDesc(*(GetBlobDesc4BnInOp(input_bns().Get(i)))).ByteSizeOfBlobBody();
  }
  CHECK_EQ(offset, RtBlobDesc(*GetBlobDesc4BnInOp(SoleObn())).ByteSizeOfBlobBody());
}

LogicalBlobId ReduceConcatOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name("out");
  return ret;
}

REGISTER_OP(OperatorConf::kReduceConcatConf, ReduceConcatOp);

}  // namespace oneflow
