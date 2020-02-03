#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class SquareSumOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SquareSumOp);
  SquareSumOp() = default;
  ~SquareSumOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void SquareSumOp::InitFromOpConf() {
  CHECK(op_conf().has_square_sum_conf());
  EnrollInputBn("x");
  EnrollOutputBn("y");
}

const PbMessage& SquareSumOp::GetCustomizedConf() const { return op_conf().square_sum_conf(); }

Maybe<void> SquareSumOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* x = GetBlobDesc4BnInOp("x");
  BlobDesc* y = GetBlobDesc4BnInOp("y");
  y->mut_shape() = Shape({1});
  y->set_data_type(x->data_type());
  return Maybe<void>::Ok();
}

Maybe<void> SquareSumOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("y")->clear_value();
  return Maybe<void>::Ok();
}

Maybe<void> SquareSumOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t num_x_axes = JUST(LogicalBlobDesc4Ibn("x"))->shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, num_x_axes) {
    SbpSignatureBuilder().Split("x", i).PartialSum("y").Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kSquareSumConf, SquareSumOp);

}  // namespace oneflow
