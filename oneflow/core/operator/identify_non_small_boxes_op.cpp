#include "oneflow/core/operator/operator.h"

namespace oneflow {

class IdentifyNonSmallBoxesOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdentifyNonSmallBoxesOp);
  IdentifyNonSmallBoxesOp() = default;
  ~IdentifyNonSmallBoxesOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_identify_non_small_boxes_conf());
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false);
  }
  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().identify_non_small_boxes_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    // input
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    CHECK_EQ_OR_RETURN(in->shape().NumAxes(), 2);
    CHECK_EQ_OR_RETURN(in->shape().At(1), 4);
    // output
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->mut_shape() = Shape({in->shape().At(0)});
    out->set_data_type(DataType::kInt8);
    out->set_is_dynamic(in->is_dynamic());

    return Maybe<void>::Ok();
  }

 private:
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf* kernel_conf, const OpContext*) const override {
    kernel_conf->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
  }
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder().Split("in", 0).Split("out", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kIdentifyNonSmallBoxesConf, IdentifyNonSmallBoxesOp);

}  // namespace oneflow
