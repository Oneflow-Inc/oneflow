#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NonMaximumSuppressionOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonMaximumSuppressionOp);
  NonMaximumSuppressionOp() = default;
  ~NonMaximumSuppressionOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().non_maximum_suppression_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf*, const OpContext*) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
};

void NonMaximumSuppressionOp::InitFromOpConf() {
  CHECK(op_conf().has_non_maximum_suppression_conf());
  EnrollInputBn("in");
  EnrollInputBn("probs");
  EnrollOutputBn("out");
  EnrollTmpBn("fw_tmp");
}

Maybe<void> NonMaximumSuppressionOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int64_t num_boxes = in_blob_desc->shape().At(1);  //(b, num_box, 4)
  int64_t blocks =
      static_cast<int64_t>(std::ceil(num_boxes * 1.0f / GetSizeOfDataType(DataType::kInt64) * 8));
  // fw_tmp
  BlobDesc* fw_tmp_blob_desc = GetBlobDesc4BnInOp("fw_tmp");
  fw_tmp_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0), num_boxes * blocks});
  fw_tmp_blob_desc->set_data_type(DataType::kInt64);
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0), num_boxes});
  out_blob_desc->set_data_type(DataType::kInt8);
  return Maybe<void>::Ok();
}

void NonMaximumSuppressionOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf, const OpContext* op_ctx) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
}

Maybe<void> NonMaximumSuppressionOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  // error!
  SbpSignatureBuilder()
      .Split("in", 0)
      .Split("probs", 0)
      .Split("out", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kNonMaximumSuppressionConf, NonMaximumSuppressionOp);

}  // namespace oneflow
