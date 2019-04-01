#include "oneflow/core/operator/cast_op.h"

namespace oneflow {

namespace {

class CastSbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastSbpSignature);
  ~CastSbpSignature() override = default;

  CastSbpSignature(const Operator* op) : ParallelSbpSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": (A,) -> (A,)"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const SbpInferHint& in_sbp_infer_hint = SbpInferHint4BnInOp("in");
    if (in_sbp_infer_hint.sbp_parallel().has_split_parallel() == false
        && in_sbp_infer_hint.parallel_num() != parallel_desc.parallel_num()) {
      return MakeSbpSigMatchParallelNumError(parallel_desc.parallel_num(),
                                             in_sbp_infer_hint.parallel_num());
    }
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["in"] = SbpInferHint4BnInOp("in").sbp_parallel();
    (*bn2sbp)["out"] = SbpInferHint4BnInOp("in").sbp_parallel();
  }
};

}  // namespace

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
  op_parallel_signatures->emplace_back(new CastSbpSignature(this));
}

REGISTER_OP(OperatorConf::kCastConf, CastOp);

}  // namespace oneflow
