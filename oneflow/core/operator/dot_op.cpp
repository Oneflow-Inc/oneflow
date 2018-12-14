#include "oneflow/core/operator/dot_op.h"
namespace oneflow {

void DotOp::InitFromOpConf() {
  CHECK(op_conf().has_dot_conf());

  EnrollInputBn("in");
  EnrollInputBn("weight");
  EnrollDataTmpBn("tmp");
  EnrollDataTmpBn("tmp_storage");
  EnrollConstBufBn("diff_multiplier");
  EnrollOutputBn("out");
  if (op_conf().dot_conf().has_bias()) { EnrollInputBn("bias"); }
}

const PbMessage& DotOp::GetCustomizedConf() const { return op_conf().dot_conf(); }

void DotOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext* parallel_ctx) const {
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  DataType data_type = in_blob_desc->data_type();
  int64_t piece_size = in_blob_desc->shape().At(0);
  int64_t dim = in_blob_desc->shape().Count(1);
  Shape temp_shape = Shape({piece_size, dim});
  CHECK_EQ(data_type, Global<JobDesc>::Get()->DefaultDataType());
  CHECK_EQ(data_type, weight_blob_desc->data_type());
  CHECK_EQ(piece_size, weight_blob_desc->shape().At(0));
  CHECK_EQ(dim, weight_blob_desc->shape().Count(1));
  // tmp & tmp storage
  BlobDesc* tmp_blob_desc = GetBlobDesc4BnInOp("tmp");
  tmp_blob_desc->set_data_type(data_type);
  tmp_blob_desc->mut_shape() = temp_shape;
  BlobDesc* tmp_storage_blob_desc = GetBlobDesc4BnInOp("tmp_storage");
  tmp_storage_blob_desc->set_data_type(data_type);
  tmp_storage_blob_desc->mut_shape() = temp_shape;
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape({piece_size, 1});
  out_blob_desc->set_has_instance_shape_field(false);
  // diff_multiplier
  BlobDesc* diff_mul_blob_desc = GetBlobDesc4BnInOp("diff_multiplier");
  diff_mul_blob_desc->set_data_type(in_blob_desc->data_type());
  diff_mul_blob_desc->mut_shape() = Shape({1, dim});
}

REGISTER_OP(OperatorConf::kDotConf, DotOp);

}  // namespace oneflow
