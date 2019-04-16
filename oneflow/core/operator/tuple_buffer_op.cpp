#include "oneflow/core/operator/tuple_buffer_op.h"

namespace oneflow {

namespace {

class TupleBufferOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TupleBufferOpParallelSignature);
  ~TupleBufferOpParallelSignature() override = default;

  explicit TupleBufferOpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": A -> A"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const auto& ibn = op().input_bns().Get(0);
    if (parallel_desc.parallel_num() != SbpInferHint4BnInOp(ibn).parallel_num()) {
      return MakeOpParallelMatchParallelNumError(parallel_desc.parallel_num(),
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

void TupleBufferOp::InitFromOpConf() {
  CHECK(op_conf().has_tuple_buffer_conf());
  const int32_t in_size = op_conf().tuple_buffer_conf().in_size();
  const int32_t out_size = op_conf().tuple_buffer_conf().out_size();
  CHECK_GT(in_size, 0);
  CHECK_EQ(in_size, out_size);
  EnrollRepeatedInputBn("in", in_size);
  EnrollRepeatedOutputBn("out", out_size);
}

const PbMessage& TupleBufferOp::GetCustomizedConf() const { return op_conf().tuple_buffer_conf(); }

void TupleBufferOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const int32_t bn_size = op_conf().tuple_buffer_conf().in_size();
  FOR_RANGE(int32_t, i, 0, bn_size) {
    *GetBlobDesc4BnInOp(output_bns().Get(i)) = *GetBlobDesc4BnInOp(input_bns().Get(i));
  }
}

void TupleBufferOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new TupleBufferOpParallelSignature(this));
}

REGISTER_OP(OperatorConf::kTupleBufferConf, TupleBufferOp);

}  // namespace oneflow
