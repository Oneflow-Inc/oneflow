#include "oneflow/core/operator/gelu_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class GeluOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluOp);
  GeluOp() = default;
  ~GeluOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_gelu_conf());

    EnrollInputBn("in");
    EnrollOutputBn("out");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().gelu_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kGeluConf, GeluOp);

}  // namespace oneflow
