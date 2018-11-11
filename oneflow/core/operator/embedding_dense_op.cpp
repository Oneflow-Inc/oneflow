#include "oneflow/core/operator/embedding_dense_op.h"

namespace oneflow {

void EmbeddingDenseOp::InitFromOpConf() {
  CHECK(op_conf().has_embedding_dense_conf());
  EnrollInputBn("ids", false);
  EnrollOutputBn("out");
  EnrollModelBn("weight");
}

const PbMessage& EmbeddingDenseOp::GetCustomizedConf() const {
  return op_conf().embedding_dense_conf();
}

void EmbeddingDenseOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const EmbeddingDenseOpConf& conf = op_conf().embedding_dense_conf();
  // input: ids [D0, ...]
  CHECK_GT(conf.units(), 0);
  CHECK_GT(conf.table_size(), 0);
  const BlobDesc* ids_blob_desc = GetBlobDesc4BnInOp("ids");
  CHECK_EQ(ids_blob_desc->data_type(), DataType::kInt32);
  // output: out [D0, ..., units]
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *ids_blob_desc;
  out_blob_desc->set_data_type(conf.data_type());
  std::vector<int64_t> out_shape_dim_vec = ids_blob_desc->shape().dim_vec();
  out_shape_dim_vec.push_back(conf.units());
  out_blob_desc->mut_shape() = Shape(out_shape_dim_vec);
  // model: weight [table_size, weight]
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  weight_blob_desc->mut_shape() = Shape({conf.table_size(), conf.units()});
  weight_blob_desc->set_data_type(conf.data_type());
}

REGISTER_OP(OperatorConf::kEmbeddingDenseConf, EmbeddingDenseOp);

}  // namespace oneflow
