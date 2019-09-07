#include "oneflow/core/operator/embedding_lookup_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void EmbeddingLookupOp::InitFromOpConf() {
  CHECK(op_conf().has_embedding_lookup_conf());

  EnrollInputBn("ids", false);
  EnrollOutputBn("out");
  EnrollTmpBn("weight");
}

const PbMessage& EmbeddingLookupOp::GetCustomizedConf() const {
  return op_conf().embedding_lookup_conf();
}

Maybe<void> EmbeddingLookupOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const EmbeddingLookupOpConf& conf = op_conf().embedding_lookup_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("ids");
  CHECK_EQ_OR_RETURN(in_blob_desc->data_type(), DataType::kInt32);
  int32_t units = conf.units();
  int32_t table_size = conf.table_size();
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(units, parallel_ctx->parallel_num());
    units = splitter.At(parallel_ctx->parallel_id()).size();
  }
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->set_data_type(GlobalJobDesc().DefaultDataType());
  out_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0), units});

  // weight
  GetBlobDesc4BnInOp("weight")->mut_shape() = Shape({table_size, units});
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kEmbeddingLookupConf, EmbeddingLookupOp);

}  // namespace oneflow
