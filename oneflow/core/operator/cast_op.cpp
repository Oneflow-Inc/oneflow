#include "oneflow/core/operator/cast_op.h"

namespace oneflow {

void CastOp::InitFromOpConf() {
  CHECK(op_conf().has_cast_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& CastOp::GetCustomizedConf() const { return op_conf().cast_conf(); }

void CastOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->set_data_type(op_conf().cast_conf().data_type());
}

void CastOp::FixInputOutputSbpParallel(
    const std::function<SbpParallel*(const std::string&)>& SbpParallel4BnInOp) const {
  if (SbpParallel4BnInOp("out")->has_partial_sum_parallel()) {
    SbpParallel4BnInOp("in")->mutable_broadcast_parallel();
    SbpParallel4BnInOp("out")->mutable_broadcast_parallel();
  }
}

void CastOp::GetSbpSignatures(
    std::vector<std::unique_ptr<const SbpSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(MakeIdentitySbpSignature(this));
}

REGISTER_OP(OperatorConf::kCastConf, CastOp);

}  // namespace oneflow
