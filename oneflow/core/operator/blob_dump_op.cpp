#include "oneflow/core/operator/blob_dump_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void BlobDumpOp::InitFromOpConf() {
  CHECK(op_conf().has_blob_dump_conf());
  EnrollInputBn("in", false);
}

const PbMessage& BlobDumpOp::GetCustomizedConf() const { return op_conf().blob_dump_conf(); }

void BlobDumpOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  (*sbp_signature->mutable_bn_in_op2sbp_parallel())["in"] = SbpInferHint4Ibn("in").sbp_parallel();
}

REGISTER_CPU_OP(OperatorConf::kBlobDumpConf, BlobDumpOp);

}  // namespace oneflow
