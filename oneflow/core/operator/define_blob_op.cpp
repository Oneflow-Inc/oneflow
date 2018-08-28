#include "oneflow/core/operator/define_blob_op.h"

namespace oneflow {

void DefineBlobOp::InitFromOpConf() {
  CHECK(op_conf().has_define_blob_conf());
  EnrollOutputBn("name", false);
}

const PbMessage& DefineBlobOp::GetCustomizedConf() const { return op_conf().define_blob_conf(); }

void DefineBlobOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("name");
  const DefineBlobConf& conf = op_conf().define_blob_conf();
  out_blob_desc->mut_shape() = Shape(conf.shape());
  out_blob_desc->set_data_type(conf.data_type());
  out_blob_desc->set_has_data_id_field(false);
  out_blob_desc->set_has_col_num_field(false);
  out_blob_desc->set_max_col_num(1);
}

REGISTER_OP(OperatorConf::kDefineBlobConf, DefineBlobOp);

}  // namespace oneflow
