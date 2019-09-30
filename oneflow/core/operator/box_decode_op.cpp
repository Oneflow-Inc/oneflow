#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BoxDecodeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxDecodeOp);
  BoxDecodeOp() = default;
  ~BoxDecodeOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_box_decode_conf());
    EnrollInputBn("ref_boxes", false);
    EnrollInputBn("boxes_delta", false);
    EnrollOutputBn("boxes", false);
  }
  const PbMessage& GetCustomizedConf() const override { return this->op_conf().box_decode_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    // input: ref_boxes (N, 4)
    const BlobDesc* ref_boxes = GetBlobDesc4BnInOp("ref_boxes");
    CHECK_EQ_OR_RETURN(ref_boxes->shape().NumAxes(), 2);
    CHECK_EQ_OR_RETURN(ref_boxes->shape().At(1), 4);
    // input: boxes_delta (N, 4)
    const BlobDesc* boxes_delta = GetBlobDesc4BnInOp("boxes_delta");
    CHECK_EQ_OR_RETURN(boxes_delta->shape().NumAxes(), 2);
    CHECK_EQ_OR_RETURN(boxes_delta->shape().At(1), 4);
    CHECK_EQ_OR_RETURN(ref_boxes->shape(), boxes_delta->shape());
    // output: boxes (N, 4)
    BlobDesc* boxes = GetBlobDesc4BnInOp("boxes");
    boxes->mut_shape() = ref_boxes->shape();
    boxes->set_data_type(ref_boxes->data_type());
    CHECK_EQ(ref_boxes->is_dynamic(), boxes_delta->is_dynamic());
    boxes->set_is_dynamic(ref_boxes->is_dynamic());

    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("boxes") = *BatchAxis4BnInOp("ref_boxes");
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kBoxDecodeConf, BoxDecodeOp);

}  // namespace oneflow
