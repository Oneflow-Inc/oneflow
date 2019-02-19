#include "oneflow/core/operator/reshape_op.h"

namespace oneflow {

void ReshapeOp::InitFromOpConf() {
  CHECK(op_conf().has_reshape_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ReshapeOp::GetCustomizedConf() const { return op_conf().reshape_conf(); }

void ReshapeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *out_blob_desc = *in_blob_desc;

  CHECK(!in_blob_desc->has_instance_shape_field());
  const ReshapeOpConf& conf = op_conf().reshape_conf();

  std::vector<int64_t> dim_vec;
  if (!conf.has_dim0_in_shape()) { dim_vec.push_back(in_blob_desc->shape().At(0)); }
  for (int32_t i = 0; i < conf.shape().dim_size(); ++i) { dim_vec.push_back(conf.shape().dim(i)); }
  int32_t dim_cnt_need_infer = 0;
  int32_t dim_index_need_infer = -1;
  int64_t elem_cnt = 1;
  for (int32_t i = 0; i < dim_vec.size(); ++i) {
    if (dim_vec[i] == -1) {
      ++dim_cnt_need_infer;
      dim_index_need_infer = i;
    } else {
      elem_cnt *= dim_vec[i];
    }
  }
  CHECK_LE(dim_cnt_need_infer, 1);
  if (dim_cnt_need_infer == 1) {
    dim_vec[dim_index_need_infer] = in_blob_desc->shape().elem_cnt() / elem_cnt;
  }
  out_blob_desc->mut_shape() = Shape(dim_vec);
  CHECK_EQ(out_blob_desc->shape().elem_cnt(), in_blob_desc->shape().elem_cnt());
}

REGISTER_OP(OperatorConf::kReshapeConf, ReshapeOp);

}  // namespace oneflow
