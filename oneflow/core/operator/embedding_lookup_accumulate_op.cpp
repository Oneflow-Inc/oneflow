#include "oneflow/core/operator/embedding_lookup_accumulate_op.h"

namespace oneflow {

void EmbeddingLookupAccumulateOp::InitFromOpConf() {
  CHECK(op_conf().has_accumulate_conf());

  EnrollInputBn("one_ids", false);
  EnrollInputBn("one_val", false);
  EnrollOutputBn("acc_ids", false);
  EnrollOutputBn("acc_val", false);
}

void EmbeddingLookupAccumulateOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // use vars
  const int64_t num_of_piece_in_batch = Global<JobDesc>::Get()->NumOfPiecesInBatch();
  const BlobDesc* ids_blob_desc = GetBlobDesc4BnInOp("one_ids");
  const Shape ids_shape = ids_blob_desc->shape();
  const BlobDesc* val_blob_desc = GetBlobDesc4BnInOp("one_val");
  const Shape val_shape = val_blob_desc->shape();

  // acc_ids
  BlobDesc* acc_ids_blob_desc = GetBlobDesc4BnInOp("acc_ids");
  *acc_ids_blob_desc = *ids_blob_desc;
  std::vector<int64_t> acc_ids_dim_vec;
  CHECK(acc_ids_dim_vec.empty());
  acc_ids_dim_vec.push_back((ids_shape.At(0) * num_of_piece_in_batch));
  FOR_RANGE(int64_t, i, 1, ids_shape.NumAxes()) { acc_ids_dim_vec.push_back(ids_shape.At(i)); }
  acc_ids_blob_desc->mut_shape() = Shape(acc_ids_dim_vec);

  // acc_val
  BlobDesc* acc_val_blob_desc = GetBlobDesc4BnInOp("acc_val");
  *acc_val_blob_desc = *val_blob_desc;
  std::vector<int64_t> acc_val_dim_vec;
  CHECK(acc_val_dim_vec.empty());
  acc_val_dim_vec.push_back((val_shape.At(0) * num_of_piece_in_batch));
  FOR_RANGE(int64_t, i, 1, val_shape.NumAxes()) { acc_val_dim_vec.push_back(val_shape.At(i)); }
  acc_val_blob_desc->mut_shape() = Shape(acc_val_dim_vec);
}

const PbMessage& EmbeddingLookupAccumulateOp::GetCustomizedConf() const {
  return op_conf().embedding_lookup_accumulate_conf();
}

REGISTER_OP(OperatorConf::kEmbeddingLookupAccumulateConf, EmbeddingLookupAccumulateOp);

}  // namespace oneflow
