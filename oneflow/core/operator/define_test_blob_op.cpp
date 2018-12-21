#include "oneflow/core/operator/define_test_blob_op.h"

namespace oneflow {

void DefineTestBlobOp::InitFromOpConf() {
  CHECK(op_conf().has_define_test_blob_conf());
  EnrollOutputBn("out", op_conf().define_test_blob_conf().has_diff());
}

const PbMessage& DefineTestBlobOp::GetCustomizedConf() const {
  return op_conf().define_test_blob_conf();
}

void DefineTestBlobOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  const DefineTestBlobOpConf& conf = op_conf().define_test_blob_conf();
  Shape shape(conf.shape());
  out_blob_desc->mut_shape() = shape;
  out_blob_desc->set_data_type(conf.data_type());
  out_blob_desc->set_has_data_id_field(Global<JobDesc>::Get()->SizeOfOneDataId() > 0);
  out_blob_desc->set_has_col_num_field(false);
  out_blob_desc->set_has_dim0_valid_num_field(conf.has_dim0_valid_num());
  out_blob_desc->set_has_dim1_valid_num_field(conf.has_dim1_valid_num());
  out_blob_desc->set_has_dim2_valid_num_field(conf.has_dim2_valid_num());
  out_blob_desc->set_has_record_id_in_device_piece_field(!conf.record_id_in_device_piece().empty());
  out_blob_desc->set_max_col_num(1);
  if (conf.has_dim0_inner_shape()) {
    out_blob_desc->mut_dim0_inner_shape() = Shape(conf.dim0_inner_shape());
  }
  if (conf.has_dim0_valid_num()) { CHECK(conf.has_dim0_inner_shape()); }
}

REGISTER_OP(OperatorConf::kDefineTestBlobConf, DefineTestBlobOp);

}  // namespace oneflow
