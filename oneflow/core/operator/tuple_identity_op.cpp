#include "oneflow/core/operator/tuple_identity_op.h"

namespace oneflow {

class TupleIdentityOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TupleIdentityOp);
  TupleIdentityOp() = default;
  ~TupleIdentityOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().tuple_identity_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  void GetOpParallelSignatures(
      std::vector<std::unique_ptr<const OpParallelSignature>>*) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
};

namespace {

class TupleIdentityOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TupleIdentityOpParallelSignature);
  ~TupleIdentityOpParallelSignature() override = default;

  TupleIdentityOpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": A -> A"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    const auto& ibn = op().input_bns().Get(0);
    if (parallel_ctx->parallel_num() != SbpInferHint4BnInOp(ibn).parallel_num()) {
      return MakeOpParallelMatchParallelNumError(parallel_ctx->parallel_num(),
                                                 SbpInferHint4BnInOp(ibn).parallel_num());
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    FOR_RANGE(int32_t, i, 0, op().input_bns().size()) {
      const auto& sbp_parallel = SbpInferHint4BnInOp(op().input_bns().Get(i)).sbp_parallel();
      (*bn2sbp)[op().input_bns().Get(i)] = sbp_parallel;
      (*bn2sbp)[op().output_bns().Get(i)] = sbp_parallel;
    }
  }
};

}  // namespace

void TupleIdentityOp::InitFromOpConf() {
  CHECK(op_conf().has_tuple_identity_conf());
  int32_t in_size = op_conf().tuple_identity_conf().in_size();
  int32_t out_size = op_conf().tuple_identity_conf().out_size();
  CHECK_GT(in_size, 0);
  CHECK_EQ(in_size, out_size);
  EnrollInputBn("in", in_size);
  EnrollOutputBn("out", out_size);
}

Maybe<void> TupleIdentityOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  size_t bn_size = op_conf().tuple_identity_conf().in_size();
  FOR_RANGE(int, i, 0, bn_size) {
    *GetBlobDesc4BnInOp(output_bns().Get(i)) = *GetBlobDesc4BnInOp(input_bns().Get(i));
  }
  return Maybe<void>::Ok();
}

void TupleIdentityOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new TupleIdentityOpParallelSignature(this));
}

REGISTER_OP(OperatorConf::kTupleIdentityConf, TupleIdentityOp);

}  // namespace oneflow
