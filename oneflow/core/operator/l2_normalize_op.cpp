#include "oneflow/core/operator/operator.h"

namespace oneflow {

class L2NormalizeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L2NormalizeOp);
  L2NormalizeOp() = default;
  ~L2NormalizeOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_l2_normalize_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out");
    EnrollOutputBn("square_x_sum");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().l2_normalize_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const L2NormalizeOpConf& conf = op_conf().l2_normalize_conf();
    const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
    int32_t axis_num = in_blob_desc->shape().NumAxes();
    int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + axis_num;
    CHECK_GE_OR_RETURN(axis, 0);
    CHECK_LT_OR_RETURN(axis, axis_num);
    CHECK_GT_OR_RETURN(conf.epsilon(), 0);
    *GetBlobDesc4BnInOp("out") = *in_blob_desc;
    BlobDesc* square_x_sum_blob_desc = GetBlobDesc4BnInOp("square_x_sum");
    *square_x_sum_blob_desc = *in_blob_desc;
    square_x_sum_blob_desc->mut_shape().Set(axis, 1);
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    for (const auto& obn : output_bns()) { *BatchAxis4BnInOp(obn) = *BatchAxis4BnInOp("in"); }
    return Maybe<void>::Ok();
  }
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kL2NormalizeConf, L2NormalizeOp);

}  // namespace oneflow
