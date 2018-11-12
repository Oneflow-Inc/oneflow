#include "oneflow/core/operator/matmul_op.h"
#include "oneflow/core/common/balanced_splitter.h"
namespace oneflow {

void MatmulOp::InitFromOpConf() {
  CHECK(op_conf().has_matmul_conf());
  EnrollInputBn("a");
  EnrollInputBn("b");
  EnrollOutputBn("out");
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
    CHECK_EQ(a_blob_desc->has_dim0_valid_num_field(), false);
    CHECK_EQ(a_blob_desc->has_dim1_valid_num_field(), false);
    CHECK_EQ(a_blob_desc->has_dim2_valid_num_field(), false);
  }
  CHECK_EQ(b_blob_desc->has_dim0_valid_num_field(), false);
  CHECK_EQ(b_blob_desc->has_dim1_valid_num_field(), false);
  CHECK_EQ(b_blob_desc->has_dim2_valid_num_field(), false);
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *a_blob_desc;
  std::vector<int64_t> a_dim_vec = a_blob_desc->shape().dim_vec();
  a_dim_vec.pop_back();
  a_dim_vec.pop_back();
  std::vector<int64_t> b_dim_vec = b_blob_desc->shape().dim_vec();
  b_dim_vec.pop_back();
  b_dim_vec.pop_back();
  CHECK(a_dim_vec == b_dim_vec);
  int64_t a_dim_index = conf.transpose_a() ? num_axes - 1 : num_axes - 2;
  out_blob_desc->mut_shape().Set(num_axes - 2, a_blob_desc->shape().At(a_dim_index));
  int64_t b_dim_index = conf.transpose_b() ? num_axes - 2 : num_axes - 1;
  out_blob_desc->mut_shape().Set(num_axes - 1, b_blob_desc->shape().At(b_dim_index));
}

REGISTER_OP(OperatorConf::kMatmulConf, MatmulOp);

}  // namespace oneflow
