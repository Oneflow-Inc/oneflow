#include "oneflow/core/operator/lookup_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void LookupOp::InitFromOpConf() {
  CHECK(op_conf().has_lookup_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollModelBn("weight");
}

const PbMessage& LookupOp::GetCustomizedConf() const {
  return op_conf().lookup_conf();
}

void LookupOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // useful vars
  const LookupOpConf& conf = op_conf().lookup_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());
  int32_t units = conf.units();
  int32_t inmax = conf.inmax();
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(units, parallel_ctx->parallel_num());
    units = splitter.At(parallel_ctx->parallel_id()).size();
  }
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0), units});

  // weight
  GetBlobDesc4BnInOp("weight")->mut_shape() = Shape({inmax, units});
}

REGISTER_OP(OperatorConf::kLookupConf, LookupOp);

}  // namespace oneflow
