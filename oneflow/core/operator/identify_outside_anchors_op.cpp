#include "oneflow/core/operator/operator.h"

namespace oneflow {

class IdentifyOutsideAnchorsOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdentifyOutsideAnchorsOp);
  IdentifyOutsideAnchorsOp() = default;
  ~IdentifyOutsideAnchorsOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().identify_outside_anchors_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf*, const OpContext*) const override;

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("anchors");
    return Maybe<void>::Ok();
  }
};

void IdentifyOutsideAnchorsOp::InitFromOpConf() {
  CHECK(op_conf().has_identify_outside_anchors_conf());

  EnrollInputBn("anchors", false);
  EnrollInputBn("image_size", false);
  // Element which map to anchor outside image equal to 1, otherwise 0
  EnrollOutputBn("out", false);
}

Maybe<void> IdentifyOutsideAnchorsOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // input: anchors (R, 4)
  const BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  CHECK_EQ_OR_RETURN(anchors_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(anchors_blob_desc->shape().At(1), 4);
  // input: image_size (2)
  const BlobDesc* image_size_blob_desc = GetBlobDesc4BnInOp("image_size");
  CHECK_EQ_OR_RETURN(image_size_blob_desc->shape().NumAxes(), 1);
  CHECK_EQ_OR_RETURN(image_size_blob_desc->shape().At(0), 2);
  CHECK_EQ_OR_RETURN(image_size_blob_desc->data_type(), DataType::kInt32);

  // output: out (R, )
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->set_data_type(DataType::kInt8);
  out_blob_desc->mut_shape() = Shape({anchors_blob_desc->shape().At(0)});
  return Maybe<void>::Ok();
}

void IdentifyOutsideAnchorsOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf, const OpContext* op_ctx) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("anchors")->data_type());
}

REGISTER_OP(OperatorConf::kIdentifyOutsideAnchorsConf, IdentifyOutsideAnchorsOp);

}  // namespace oneflow
