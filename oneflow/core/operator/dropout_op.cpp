#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class DropoutOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DropoutOp);
  DropoutOp() = default;
  ~DropoutOp() = default;

  void InitFromOpConf() override {
    CHECK_GT(op_conf().dropout_conf().scale(), 1);
    EnrollInputBn("in");
    EnrollInputBn("mask", false);
    EnrollOutputBn("out");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().dropout_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("mask")->shape(), GetBlobDesc4BnInOp("in")->shape());
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("mask")->data_type(), DataType::kInt8);
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp(SoleObn()) = *BatchAxis4BnInOp("in");
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes())
        .Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

class DropoutGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DropoutGradOp);
  DropoutGradOp() = default;
  ~DropoutGradOp() = default;

  void InitFromOpConf() override {
    EnrollInputBn("dy");
    EnrollInputBn("mask");
    EnrollOutputBn("dx");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().dropout_grad_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    BlobDesc* dy_desc = GetBlobDesc4BnInOp("dy");
    *GetBlobDesc4BnInOp("dx") = *dy_desc;
    CHECK_EQ_OR_RETURN(dy_desc->shape(), GetBlobDesc4BnInOp("mask")->shape());
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn("dy"))->shape().NumAxes())
        .Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

class RandomMaskLikeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomMaskLikeOp);
  RandomMaskLikeOp() = default;
  ~RandomMaskLikeOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().random_mask_like_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp(SoleObn()) = *BatchAxis4BnInOp(SoleIbn());
    return Maybe<void>::Ok();
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void RandomMaskLikeOp::InitFromOpConf() {
  if (op_conf().random_mask_like_conf().has_noise_shape()) { TODO(); }
  double rate = op_conf().random_mask_like_conf().rate();
  CHECK_GE(rate, 0);
  CHECK_LT(rate, 1);
  EnrollInputBn("like", false)->set_use_header_only(true);
  EnrollTmpBn("random_tmp");
  EnrollOutputBn("out", false);
}

Maybe<void> RandomMaskLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // CHECK_EQ(op_conf().random_mask_like_conf().noise_shape().dim_size(),
  //          GetBlobDesc4BnInOp("in")->shape().NumAxes());
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->CopyMetaFrom(*GetBlobDesc4BnInOp("like"));
  out->set_data_type(DataType::kInt8);
  BlobDesc* random_tmp = GetBlobDesc4BnInOp("random_tmp");
  random_tmp->CopyMetaFrom(*GetBlobDesc4BnInOp("like"));
  random_tmp->set_data_type(DataType::kFloat);
  return Maybe<void>::Ok();
}

Maybe<void> RandomMaskLikeOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn(SoleIbn()))->shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kDropoutConf, DropoutOp);
REGISTER_OP(OperatorConf::kDropoutGradConf, DropoutGradOp);
REGISTER_OP(OperatorConf::kRandomMaskLikeConf, RandomMaskLikeOp);

}  // namespace oneflow
