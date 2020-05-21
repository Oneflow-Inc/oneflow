#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class MultiplyOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultiplyOp);
  MultiplyOp() = default;
  ~MultiplyOp() override = default;
  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void MultiplyOp::InitFromOpConf() {
  CHECK(op_conf().has_multiply_conf());
  EnrollInputBn("in_0");
  EnrollInputBn("in_1");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in_0");
}

const PbMessage& MultiplyOp::GetCustomizedConf() const { return op_conf().multiply_conf(); }

Maybe<void> MultiplyOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp("in_0");
  BlobDesc* in_1_blob_desc = GetBlobDesc4BnInOp("in_1");
  CHECK_EQ_OR_RETURN(in_0_blob_desc->shape(), in_1_blob_desc->shape());
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_0_blob_desc;
  return Maybe<void>::Ok();
}

Maybe<void> MultiplyOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(
          JUST(LogicalBlobDesc4Ibn(input_bns().Get(0)))->shape().NumAxes())
      .Build(sbp_sig_list);
  SbpSignatureBuilder().Broadcast("in_0").PartialSum("in_1").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder().PartialSum("in_0").Broadcast("in_1").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kMultiplyConf, MultiplyOp);

}  // namespace oneflow
