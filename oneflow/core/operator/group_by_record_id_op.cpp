#include "oneflow/core/operator/group_by_record_id_op.h"

namespace oneflow {

void GroupByRecordIdOp::InitFromOpConf() {
  CHECK(op_conf().has_group_by_record_id_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& GroupByRecordIdOp::GetCustomizedConf() const {
  return op_conf().group_by_record_id_conf();
}

void GroupByRecordIdOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // input: in (R, ...)
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GE(in->shape().NumAxes(), 1);
  CHECK(in->has_record_id_in_device_piece_field());
  CHECK(!in->has_dim2_valid_num_field());

  // output: out(N, R, ...)
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  std::vector<int64_t> dim_vec;
  // TODO: device piece size
  dim_vec.push_back(1);
  dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin(), in->shape().dim_vec().cend());
  out->mut_shape() = Shape(dim_vec);
  out->set_data_type(in->data_type());
  out->set_has_dim1_valid_num_field(true);
  out->set_has_dim2_valid_num_field(in->has_dim1_valid_num_field());
}

REGISTER_CPU_OP(OperatorConf::kGroupByRecordIdConf, GroupByRecordIdOp);

}  // namespace oneflow
