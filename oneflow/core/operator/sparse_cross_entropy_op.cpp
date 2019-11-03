#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SparseCrossEntropyOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseCrossEntropyOp);
  SparseCrossEntropyOp() = default;
  ~SparseCrossEntropyOp() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_sparse_cross_entropy_conf());
    EnrollInputBn("prediction");
    EnrollInputBn("label", false);
    EnrollOutputBn("out");
  }
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().sparse_cross_entropy_conf();
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
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
    *out_blob_desc = *pred_blob_desc;
    out_blob_desc->mut_shape() = Shape(std::vector<int64_t>(
        pred_blob_desc->shape().dim_vec().cbegin(), pred_blob_desc->shape().dim_vec().cend() - 1));
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

class SparseCrossEntropyMs1Op final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseCrossEntropyMs1Op);
  SparseCrossEntropyMs1Op() = default;
  ~SparseCrossEntropyMs1Op() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_sparse_cross_entropy_ms1_conf());
    EnrollInputBn("prediction");
    EnrollInputBn("label", false);
    EnrollOutputBn("out");
  }
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().sparse_cross_entropy_ms1_conf();
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
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
    *out_blob_desc = *pred_blob_desc;
    out_blob_desc->mut_shape() = Shape(std::vector<int64_t>(
        pred_blob_desc->shape().dim_vec().cbegin(), pred_blob_desc->shape().dim_vec().cend() - 1));
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("prediction", 1)
        .Broadcast("label")
        .PartialSum("out")
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    const int64_t dim = op_conf().sparse_cross_entropy_ms1_conf().depth();
    CHECK_GE(dim, parallel_ctx->parallel_num());
    BalancedSplitter bs(dim, parallel_ctx->parallel_num());
    kernel_conf->mutable_sparse_cross_entropy_conf()->set_lower_bound(
        bs.At(parallel_ctx->parallel_id()).begin());
  }
};

REGISTER_OP(OperatorConf::kSparseCrossEntropyConf, SparseCrossEntropyOp);
REGISTER_OP(OperatorConf::kSparseCrossEntropyMs1Conf, SparseCrossEntropyMs1Op);

}  // namespace oneflow
