#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

class IndexedSlicesSGDUpdateOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IndexedSlicesSGDUpdateOp);
  IndexedSlicesSGDUpdateOp() = default;
  ~IndexedSlicesSGDUpdateOp() override = default;

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

void IndexedSlicesSGDUpdateOp::InitFromOpConf() {
  CHECK(op_conf().has_indexed_slices_sgd_update_conf());
  const IndexedSlicesSGDUpdateOpConf& conf = op_conf().indexed_slices_sgd_update_conf();
  EnrollInputBn("indices", false);
  EnrollInputBn("updates", false);
  EnrollInputBn("ref", false)->set_is_mutable(true);
  if (conf.has_momentum()) { EnrollInputBn("momentum", false)->set_is_mutable(true); }
  EnrollInputBn("learning_rate", false);
}

const PbMessage& IndexedSlicesSGDUpdateOp::GetCustomizedConf() const {
  return op_conf().indexed_slices_sgd_update_conf();
}

Maybe<void> IndexedSlicesSGDUpdateOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const IndexedSlicesSGDUpdateOpConf& conf = op_conf().indexed_slices_sgd_update_conf();
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  const BlobDesc* updates = GetBlobDesc4BnInOp("updates");
  OF_CHECK(IsIndexDataType(indices->data_type()));
  const int64_t num_indices_axes = indices->shape().NumAxes();
  const int64_t num_updates_axes = updates->shape().NumAxes();
  OF_CHECK_GE(num_updates_axes, num_indices_axes);
  FOR_RANGE(int64_t, i, 0, num_indices_axes) {
    OF_CHECK_EQ(updates->shape().At(i), indices->shape().At(i));
  }
  const BlobDesc* ref = GetBlobDesc4BnInOp("ref");
  OF_CHECK_EQ(ref->data_type(), updates->data_type());
  const int64_t num_ref_axes = ref->shape().NumAxes();
  OF_CHECK_EQ(num_ref_axes, num_updates_axes - num_indices_axes + 1);
  FOR_RANGE(int64_t, i, 1, num_ref_axes) {
    OF_CHECK_EQ(ref->shape().At(i), updates->shape().At(num_indices_axes + i));
  }
  if (conf.has_momentum()) {
    const BlobDesc* momentum = GetBlobDesc4BnInOp("momentum");
    OF_CHECK(momentum == ref);
  } else {
    OF_CHECK_EQ(conf.beta(), 0);
  }
  const BlobDesc* learning_rate = GetBlobDesc4BnInOp("learning_rate");
  OF_CHECK_EQ(learning_rate->data_type(), DataType::kFloat);
  OF_CHECK_EQ(learning_rate->shape(), Shape({1}));
  return Maybe<void>::Ok();
}

Maybe<void> IndexedSlicesSGDUpdateOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const IndexedSlicesSGDUpdateOpConf& conf = op_conf().indexed_slices_sgd_update_conf();
  const int64_t num_indices_axes = JUST(LogicalBlobDesc4Ibn("indices"))->shape().NumAxes();
  const int64_t num_ref_axes = JUST(LogicalBlobDesc4Ibn("ref"))->shape().NumAxes();
  PbRpf<std::string> state_bns;
  *state_bns.Add() = "ref";
  if (conf.has_momentum()) { *state_bns.Add() = "momentum"; }
  SbpSignatureBuilder()
      .Broadcast("learning_rate")
      .Broadcast("indices")
      .Broadcast("updates")
      .Split(state_bns, 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  FOR_RANGE(int64_t, i, 1, num_ref_axes) {
    SbpSignatureBuilder()
        .Broadcast("learning_rate")
        .Broadcast("indices")
        .Split("updates", num_indices_axes + i - 1)
        .Split(state_bns, i)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

void IndexedSlicesSGDUpdateOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext*,
    std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const {
  const BlobDesc& ref_logical_blob_desc = LogicalBlobDesc4BnInOp("ref");
  const BlobDesc* ref_blob_desc = GetBlobDesc4BnInOp("ref");
  const BlobDesc* indices_blob = GetBlobDesc4BnInOp("indices");
  const int64_t num_ref_instances = ref_logical_blob_desc.shape().At(0);
  IndexedSlicesSGDUpdateKernelConf* indexed_slices_sgd_update_conf =
      kernel_conf->mutable_indexed_slices_sgd_update_conf();
  indexed_slices_sgd_update_conf->set_indices_data_type(indices_blob->data_type());
  if (ref_blob_desc->shape().At(0) == num_ref_instances) {
    indexed_slices_sgd_update_conf->set_lower_bound(0);
    indexed_slices_sgd_update_conf->set_upper_bound(num_ref_instances);
  } else {
    BalancedSplitter bs(num_ref_instances, parallel_ctx->parallel_num());
    indexed_slices_sgd_update_conf->set_lower_bound(bs.At(parallel_ctx->parallel_id()).begin());
    indexed_slices_sgd_update_conf->set_upper_bound(bs.At(parallel_ctx->parallel_id()).end());
  }
}

REGISTER_OP(OperatorConf::kIndexedSlicesSgdUpdateConf, IndexedSlicesSGDUpdateOp);

}  // namespace oneflow
