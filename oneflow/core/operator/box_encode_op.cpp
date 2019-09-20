#include "oneflow/core/operator/operator.h"

namespace oneflow {
class BoxEncodeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxEncodeOp);
  BoxEncodeOp() = default;
  ~BoxEncodeOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return this->op_conf().box_encode_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

void BoxEncodeOp::InitFromOpConf() {
  CHECK(op_conf().has_box_encode_conf());
  EnrollInputBn("ref_boxes", false);
  EnrollInputBn("boxes", false);
  EnrollOutputBn("boxes_delta", false);
}

Maybe<void> BoxEncodeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  // input: ref_boxes (N, 4)
  const BlobDesc* ref_boxes = GetBlobDesc4BnInOp("ref_boxes");
  CHECK_EQ_OR_RETURN(ref_boxes->shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(ref_boxes->shape().At(1), 4);
  // input: boxes (N, 4)
  const BlobDesc* boxes = GetBlobDesc4BnInOp("boxes");
  CHECK_EQ_OR_RETURN(boxes->shape().NumAxes(), 2);
  CHECK_EQ_OR_RETURN(boxes->shape().At(1), 4);
  CHECK_EQ_OR_RETURN(ref_boxes->shape(), boxes->shape());
  // output: boxes_delta (N, 4)
  BlobDesc* boxes_delta = GetBlobDesc4BnInOp("boxes_delta");
  boxes_delta->mut_shape() = ref_boxes->shape();
  boxes_delta->set_data_type(ref_boxes->data_type());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBoxEncodeConf, BoxEncodeOp);

}  // namespace oneflow
