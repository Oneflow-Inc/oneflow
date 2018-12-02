#include "oneflow/core/operator/matmul_op.h"
#include "oneflow/core/common/balanced_splitter.h"
namespace oneflow {

void MatmulOp::InitFromOpConf() {
  CHECK(op_conf().has_matmul_conf());
  EnrollInputBn("a");
  EnrollInputBn("b");
  EnrollOutputBn("out");
  EnrollFwBufBn("fw_buf");
  EnrollBwBufBn("bw_buf");
}

const PbMessage& MatmulOp::GetCustomizedConf() const { return op_conf().matmul_conf(); }

void MatmulOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const MatmulOpConf& conf = op_conf().matmul_conf();
  BlobDesc* a_blob_desc = GetBlobDesc4BnInOp("a");
  BlobDesc* b_blob_desc = GetBlobDesc4BnInOp("b");
  CHECK_EQ(a_blob_desc->shape().NumAxes(), b_blob_desc->shape().NumAxes());
  CHECK_GE(a_blob_desc->shape().NumAxes(), 2);
  size_t num_axes = a_blob_desc->shape().NumAxes();
  if (conf.transpose_a()) {
    CHECK(!a_blob_desc->has_dim0_valid_num_field());
    CHECK(!a_blob_desc->has_dim1_valid_num_field());
    CHECK(!a_blob_desc->has_dim2_valid_num_field());
  }
  if (conf.transpose_b()) {
    CHECK(!b_blob_desc->has_dim0_valid_num_field());
    CHECK(!b_blob_desc->has_dim1_valid_num_field());
    CHECK(!b_blob_desc->has_dim2_valid_num_field());
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *a_blob_desc;
  FOR_RANGE(int32_t, i, 0, num_axes - 2) {
    CHECK_EQ(a_blob_desc->shape().At(i), b_blob_desc->shape().At(i));
  }
  int64_t a_dim_index = conf.transpose_a() ? num_axes - 1 : num_axes - 2;
  out_blob_desc->mut_shape().Set(num_axes - 2, a_blob_desc->shape().At(a_dim_index));
  int64_t b_dim_index = conf.transpose_b() ? num_axes - 2 : num_axes - 1;
  out_blob_desc->mut_shape().Set(num_axes - 1, b_blob_desc->shape().At(b_dim_index));
  int64_t a_mid_dim_index = conf.transpose_a() ? num_axes - 2 : num_axes - 1;
  int64_t b_mid_dim_index = conf.transpose_b() ? num_axes - 1 : num_axes - 2;
  CHECK_EQ(a_blob_desc->shape().At(a_mid_dim_index), b_blob_desc->shape().At(b_mid_dim_index));
  if (device_type() == DeviceType::kGPU && num_axes >= 3) {
    int batch_num = a_blob_desc->shape().Count(0, num_axes - 2);
    // Assume gpu address is 64 bit
    BlobDesc* fw_buf_blob_desc = GetBlobDesc4BnInOp("fw_buf");
    *fw_buf_blob_desc = *out_blob_desc;
    fw_buf_blob_desc->mut_shape() = {3 * batch_num};
    fw_buf_blob_desc->set_data_type(DataType::kInt64);
    fw_buf_blob_desc->set_has_data_id_field(false);
  }
}

void MatmulOp::InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext*) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  size_t num_axes = out_blob_desc->shape().NumAxes();
  if (device_type() == DeviceType::kGPU && num_axes >= 3) {
    BlobDesc* bw_buf_blob_desc = GetBlobDesc4BnInOp("bw_buf");
    int32_t batch_num = out_blob_desc->shape().Count(0, num_axes - 2);
    *bw_buf_blob_desc = *out_blob_desc;
    bw_buf_blob_desc->mut_shape() = {3 * batch_num};
    bw_buf_blob_desc->set_data_type(DataType::kInt64);
    bw_buf_blob_desc->set_has_data_id_field(false);
  }
}

REGISTER_OP(OperatorConf::kMatmulConf, MatmulOp);

}  // namespace oneflow
