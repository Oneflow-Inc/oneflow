#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

class IndexedSlicesNaiveMdUpdateOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IndexedSlicesNaiveMdUpdateOp);
  IndexedSlicesNaiveMdUpdateOp() = default;
  ~IndexedSlicesNaiveMdUpdateOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
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
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf*, const OpContext*,
      std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const override;
};

void IndexedSlicesNaiveMdUpdateOp::InitFromOpConf() {
  CHECK(op_conf().has_indexed_slices_naive_model_update_conf());
  EnrollInputBn("model_diff_indices", false);
  EnrollInputBn("model_diff_values", false);
  EnrollInputBn("model", false)->set_is_mutable(true);
  EnrollInputBn("learning_rate", false);
}

const PbMessage& IndexedSlicesNaiveMdUpdateOp::GetCustomizedConf() const {
  return op_conf().indexed_slices_naive_model_update_conf();
}

Maybe<void> IndexedSlicesNaiveMdUpdateOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* indices = GetBlobDesc4BnInOp("model_diff_indices");
  const BlobDesc* values = GetBlobDesc4BnInOp("model_diff_values");
  CHECK_OR_RETURN(IsIndexDataType(indices->data_type()));
  const int64_t num_indices_axes = indices->shape().NumAxes();
  const int64_t num_values_axes = values->shape().NumAxes();
  CHECK_GE_OR_RETURN(num_values_axes, num_indices_axes);
  FOR_RANGE(int64_t, i, 0, num_indices_axes) {
    CHECK_EQ_OR_RETURN(values->shape().At(i), indices->shape().At(i));
  }
  const BlobDesc* model = GetBlobDesc4BnInOp("model");
  CHECK_EQ_OR_RETURN(model->data_type(), values->data_type());
  const int64_t num_model_axes = model->shape().NumAxes();
  CHECK_EQ_OR_RETURN(num_model_axes, num_values_axes - num_indices_axes + 1);
  FOR_RANGE(int64_t, i, 1, num_model_axes) {
    CHECK_EQ_OR_RETURN(model->shape().At(i), values->shape().At(num_indices_axes + i - 1));
  }
  const BlobDesc* learning_rate = GetBlobDesc4BnInOp("learning_rate");
  CHECK_EQ_OR_RETURN(learning_rate->data_type(), DataType::kFloat);
  CHECK_EQ_OR_RETURN(learning_rate->shape(), Shape({1}));
  return Maybe<void>::Ok();
}

Maybe<void> IndexedSlicesNaiveMdUpdateOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t num_indices_axes =
      JUST(LogicalBlobDesc4Ibn("model_diff_indices"))->shape().NumAxes();
  const int64_t num_model_axes = JUST(LogicalBlobDesc4Ibn("model"))->shape().NumAxes();
  SbpSignatureBuilder()
      .Broadcast("learning_rate")
      .Broadcast("model_diff_indices")
      .Broadcast("model_diff_values")
      .Split("model", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  FOR_RANGE(int64_t, i, 1, num_model_axes) {
    SbpSignatureBuilder()
        .Broadcast("learning_rate")
        .Broadcast("model_diff_indices")
        .Split("model_diff_values", num_indices_axes + i - 1)
        .Split("model", i)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

void IndexedSlicesNaiveMdUpdateOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext*,
    std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const {
  const BlobDesc& model_logical_blob_desc = LogicalBlobDesc4BnInOp("model");
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  const BlobDesc* indices_blob = GetBlobDesc4BnInOp("model_diff_indices");
  kernel_conf->set_data_type(model_blob_desc->data_type());
  const int64_t num_model_instances = model_logical_blob_desc.shape().At(0);
  IndexedSlicesNaiveModelUpdateKernelConf* indexed_slices_naive_model_update_conf =
      kernel_conf->mutable_indexed_slices_naive_model_update_conf();
  indexed_slices_naive_model_update_conf->set_indices_data_type(indices_blob->data_type());
  if (model_blob_desc->shape().At(0) == num_model_instances) {
    indexed_slices_naive_model_update_conf->set_lower_bound(0);
    indexed_slices_naive_model_update_conf->set_upper_bound(num_model_instances);
  } else {
    BalancedSplitter bs(num_model_instances, parallel_ctx->parallel_num());
    indexed_slices_naive_model_update_conf->set_lower_bound(
        bs.At(parallel_ctx->parallel_id()).begin());
    indexed_slices_naive_model_update_conf->set_upper_bound(
        bs.At(parallel_ctx->parallel_id()).end());
  }
}

REGISTER_OP(OperatorConf::kIndexedSlicesNaiveModelUpdateConf, IndexedSlicesNaiveMdUpdateOp);

}  // namespace oneflow
