#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CalcIoUMatrixOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CalcIoUMatrixOp);
  CalcIoUMatrixOp() = default;
  ~CalcIoUMatrixOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_calc_iou_matrix_conf());
    EnrollInputBn("boxes1", false);
    EnrollInputBn("boxes2", false);
    EnrollOutputBn("iou_matrix", false);
  }
  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().calc_iou_matrix_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    // input: boxes1 (M, 4)
    const BlobDesc* boxes1 = GetBlobDesc4BnInOp("boxes1");
    CHECK_EQ_OR_RETURN(boxes1->shape().NumAxes(), 2);
    CHECK_EQ_OR_RETURN(boxes1->shape().At(1), 4);
    const int32_t num_boxes1 = boxes1->shape().At(0);
    // input: boxes2 (G, 4)
    const BlobDesc* boxes2 = GetBlobDesc4BnInOp("boxes2");
    CHECK_EQ_OR_RETURN(boxes2->shape().NumAxes(), 2);
    CHECK_EQ_OR_RETURN(boxes2->shape().At(1), 4);
    const int32_t num_boxes2 = boxes2->shape().At(0);
    // output: iou_matrix (M, G)
    BlobDesc* iou_matrix = GetBlobDesc4BnInOp("iou_matrix");
    iou_matrix->mut_shape() = Shape({num_boxes1, num_boxes2});
    iou_matrix->set_data_type(DataType::kFloat);
    iou_matrix->set_is_dynamic(boxes1->is_dynamic() || boxes2->is_dynamic());

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
        .Split("boxes1", 0)
        .Split("boxes2", 0)
        .Split("iou_matrix", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kCalcIouMatrixConf, CalcIoUMatrixOp);

}  // namespace oneflow
