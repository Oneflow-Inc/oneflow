#include "oneflow/core/operator/reduce_local_add2_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void ReduceLocalAdd2Op::InitFromOpConf() {
  CHECK(op_conf().has_reduce_local_add2_conf());
  FOR_RANGE(int32_t, i, 0, op_conf().reduce_local_add2_conf().in_num()) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  FOR_RANGE(int32_t, i, 0, op_conf().reduce_local_add2_conf().out_num()) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
}

const PbMessage& ReduceLocalAdd2Op::GetCustomizedConf() const {
  return op_conf().reduce_local_add2_conf();
}

LogicalBlobId ReduceLocalAdd2Op::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(output_bn);
  return ret;
}

void ReduceLocalAdd2Op::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  int32_t in_num = op_conf().reduce_local_add2_conf().in_num();
  CHECK_GE(in_num, 2);
  const BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  FOR_RANGE(int32_t, i, 1, in_num) {
    CHECK(*first_in_blob == *GetBlobDesc4BnInOp(input_bns().Get(i)));
  }

  int64_t model_elem_cnt = first_in_blob->shape().elem_cnt();
  int32_t out_num = op_conf().reduce_local_add2_conf().out_num();
  CHECK_EQ(0, model_elem_cnt % out_num);
  FOR_RANGE(int32_t, i, 0, out_num) {
    BlobDesc* out_blob_i = GetBlobDesc4BnInOp("out_" + std::to_string(i));
    *out_blob_i = *first_in_blob;
    out_blob_i->mut_shape() = Shape({model_elem_cnt / out_num});
  }
}

REGISTER_OP(OperatorConf::kReduceLocalAdd2Conf, ReduceLocalAdd2Op);

}  // namespace oneflow
