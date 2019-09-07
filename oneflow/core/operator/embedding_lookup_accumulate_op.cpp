#include "oneflow/core/operator/embedding_lookup_accumulate_op.h"

namespace oneflow {

void EmbeddingLookupAccumulateOp::InitFromOpConf() {
  CHECK(op_conf().has_accumulate_conf());

  EnrollInputBn("one_ids", false);
  EnrollInputBn("one_val", false);
  EnrollOutputBn("acc_ids", false);
  EnrollOutputBn("acc_val", false);
}

Maybe<void> EmbeddingLookupAccumulateOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // use vars
  const int64_t num_of_piece_in_batch = GlobalJobDesc().NumOfPiecesInBatch();
  const BlobDesc* ids_blob_desc = GetBlobDesc4BnInOp("one_ids");
  const BlobDesc* val_blob_desc = GetBlobDesc4BnInOp("one_val");

  // acc_ids
  BlobDesc* acc_ids_blob_desc = GetBlobDesc4BnInOp("acc_ids");
  *acc_ids_blob_desc = *ids_blob_desc;
  acc_ids_blob_desc->mut_shape().Set(0, (ids_blob_desc->shape().At(0)) * num_of_piece_in_batch);

  // acc_val
  BlobDesc* acc_val_blob_desc = GetBlobDesc4BnInOp("acc_val");
  *acc_val_blob_desc = *val_blob_desc;
  acc_val_blob_desc->mut_shape().Set(0, (val_blob_desc->shape().At(0)) * num_of_piece_in_batch);
  return Maybe<void>::Ok();
}

const PbMessage& EmbeddingLookupAccumulateOp::GetCustomizedConf() const {
  return op_conf().embedding_lookup_accumulate_conf();
}

REGISTER_OP(OperatorConf::kEmbeddingLookupAccumulateConf, EmbeddingLookupAccumulateOp);

}  // namespace oneflow
