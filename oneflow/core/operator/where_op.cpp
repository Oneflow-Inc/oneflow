#include "oneflow/core/operator/where_op.h"

namespace oneflow {

void WhereOp::InitFromOpConf() {
  CHECK(op_conf().has_where_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& WhereOp::GetCustomizedConf() const { return this->op_conf().where_conf(); }

void WhereOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  // input
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const int64_t elem_cnt = in->shape().elem_cnt();
  const int32_t num_axes = in->shape().NumAxes();

  // output: (elem_cnt, num_axes)
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->mut_shape() = Shape({elem_cnt, num_axes});
  out->set_data_type(DataType::kInt32);
  out->set_has_dim0_valid_num_field(true);
}

REGISTER_OP(OperatorConf::kWhereConf, WhereOp);

}  // namespace oneflow
