#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SparseSoftmaxCrossEntropyMs1Op final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseSoftmaxCrossEntropyMs1Op);
  SparseSoftmaxCrossEntropyMs1Op() = default;
  ~SparseSoftmaxCrossEntropyMs1Op() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_sparse_softmax_cross_entropy_ms1_conf());
    EnrollInputBn("prediction");
    EnrollInputBn("label", false);
    EnrollTmpBn("fw_softmax_num");
    EnrollTmpBn("fw_buf");
    EnrollOutputBn("prob");
    EnrollOutputBn("out");
  }
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().sparse_softmax_cross_entropy_ms1_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
    const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
    CHECK_OR_RETURN(IsIntegralDataType(label_blob_desc->data_type()));
    CHECK_OR_RETURN(IsFloatingDataType(pred_blob_desc->data_type()));
    CHECK_EQ_OR_RETURN(pred_blob_desc->has_data_id_field(), label_blob_desc->has_data_id_field());
    CHECK_EQ_OR_RETURN(pred_blob_desc->has_dim0_valid_num_field(),
                       label_blob_desc->has_dim0_valid_num_field());
    CHECK_EQ_OR_RETURN(pred_blob_desc->has_dim0_inner_shape(),
                       label_blob_desc->has_dim0_inner_shape());
    if (pred_blob_desc->has_dim0_inner_shape()) {
      CHECK_EQ_OR_RETURN(pred_blob_desc->dim0_inner_shape().At(0), 1);
      CHECK_EQ_OR_RETURN(pred_blob_desc->dim0_inner_shape(), label_blob_desc->dim0_inner_shape());
    }
    CHECK_GE_OR_RETURN(pred_blob_desc->shape().NumAxes(), 2);
    const int64_t num_out_axes = pred_blob_desc->shape().NumAxes() - 1;
    CHECK_GE_OR_RETURN(label_blob_desc->shape().NumAxes(), num_out_axes);
    CHECK_EQ_OR_RETURN(label_blob_desc->shape().Count(num_out_axes), 1);
    FOR_RANGE(int64_t, i, 0, num_out_axes) {
      CHECK_EQ_OR_RETURN(pred_blob_desc->shape().At(i), label_blob_desc->shape().At(i));
    }
    // 1D blob store tmp calculate result
    BlobDesc* fw_tmp_blob_desc = GetBlobDesc4BnInOp("fw_softmax_num");
    fw_tmp_blob_desc->mut_shape() = Shape({pred_blob_desc->shape().At(0)});
    fw_tmp_blob_desc->set_data_type(pred_blob_desc->data_type());
    // temp storage for RowMax etc.
    BlobDesc* fw_buf_blob_desc = GetBlobDesc4BnInOp("fw_buf");
    fw_buf_blob_desc->mut_shape() =
        Shape({static_cast<int64_t>(RtBlobDesc(*pred_blob_desc).ByteSizeOfDataContentField())});
    fw_buf_blob_desc->set_data_type(DataType::kChar);

    // prob
    BlobDesc* prob_blob_desc = GetBlobDesc4BnInOp("prob");
    prob_blob_desc->mut_shape() = Shape(pred_blob_desc->shape());
    prob_blob_desc->set_data_type(pred_blob_desc->data_type());
    // out
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
    *out_blob_desc = *pred_blob_desc;
    out_blob_desc->mut_shape() = Shape(std::vector<int64_t>(
        pred_blob_desc->shape().dim_vec().cbegin(), pred_blob_desc->shape().dim_vec().cend() - 1));
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("prob") = *BatchAxis4BnInOp("prediction");
    *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("prediction");
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("prediction", 1)
        .Broadcast("label")
        .Split("prob", 1)
        .PartialSum("out")
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kSparseSoftmaxCrossEntropyMs1Conf, SparseSoftmaxCrossEntropyMs1Op);

}  // namespace oneflow
