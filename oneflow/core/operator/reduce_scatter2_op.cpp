#include "oneflow/core/operator/reduce_scatter2_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void ReduceScatter2Op::InitFromOpConf() {
  CHECK(op_conf().has_reduce_scatter2_conf());
  EnrollInputBn("in", false);
  for (int32_t i = 0; i < op_conf().reduce_scatter2_conf().out_num(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
}

const PbMessage& ReduceScatter2Op::GetCustomizedConf() const {
  return op_conf().reduce_scatter2_conf();
}

LogicalBlobId ReduceScatter2Op::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(output_bn);
  return ret;
}

void ReduceScatter2Op::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  int32_t out_num = op_conf().reduce_scatter2_conf().out_num();
  CHECK_GE(out_num, 2);

  const BlobDesc* in_blob = GetBlobDesc4BnInOp(SoleIbn());
  int64_t model_elem_cnt = in_blob->shape().elem_cnt();
  CHECK_EQ(0, model_elem_cnt % out_num);

  for (int32_t i = 0; i < out_num; ++i) {
    BlobDesc* out_blob = GetBlobDesc4BnInOp(output_bns().Get(i));
    *out_blob = *in_blob;
    out_blob->mut_shape() = Shape({model_elem_cnt / out_num});
  }
}

REGISTER_OP(OperatorConf::kReduceScatter2Conf, ReduceScatter2Op);

}  // namespace oneflow
