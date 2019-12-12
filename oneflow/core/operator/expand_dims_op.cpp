#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ExpandDimsOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExpandDimsOp);
  ExpandDimsOp() = default;
  ~ExpandDimsOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().expand_dims_conf(); }

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

void ExpandDimsOp::InitFromOpConf() {
  CHECK(op_conf().has_expand_dims_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

Maybe<void> ExpandDimsOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  const int32_t axis = op_conf().expand_dims_conf().axis();
  DimVector dim_vec = in->shape().dim_vec();
  // do not allow expand the first dim
  CHECK_GT_OR_RETURN(axis, 0);
  CHECK_LE_OR_RETURN(axis, dim_vec.size());
  dim_vec.insert(dim_vec.begin() + axis, 1);
  *out = *in;
  out->mut_shape() = Shape(dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> ExpandDimsOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int expand_axis = op_conf().expand_dims_conf().axis();
  if (expand_axis == 0) {
    SbpSignatureBuilder().Split("in", 0).Split("out", 1).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
  } else {
    SbpSignatureBuilder().Split("in", 0).Split("out", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kExpandDimsConf, ExpandDimsOp);

}  // namespace oneflow
