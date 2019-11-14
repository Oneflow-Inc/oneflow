#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SyncDynamicResizeOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SyncDynamicResizeOp);
  SyncDynamicResizeOp() = default;
  ~SyncDynamicResizeOp() override = default;

  void InitFromOpConf() override {
    EnrollInputBn("in");
    EnrollInputBn("size", false);
    EnrollOutputBn("out");
  }
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().sync_dynamic_resize_conf();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const SyncDynamicResizeOpConf& conf = op_conf().sync_dynamic_resize_conf();
    CHECK_EQ_OR_RETURN(conf.axis(), 0);
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    CHECK_EQ_OR_RETURN(in->num_of_lod_levels(), 0);
    const BlobDesc* size = GetBlobDesc4BnInOp("size");
    CHECK_EQ_OR_RETURN(size->shape().elem_cnt(), 1);
    CHECK_EQ_OR_RETURN(size->data_type(), DataType::kInt32);
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    out->set_is_dynamic(true);
    return Maybe<void>::Ok();
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    CHECK_OR_RETURN(BatchAxis4BnInOp("in")->has_value());
    CHECK_EQ_OR_RETURN(BatchAxis4BnInOp("in")->value(), 0);
    CHECK_OR_RETURN(BatchAxis4BnInOp("size")->has_value());
    CHECK_EQ_OR_RETURN(BatchAxis4BnInOp("size")->value(), 0);
    BatchAxis4BnInOp("out")->set_value(0);
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder().Split("in", 0).Broadcast("size").Split("out", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kSyncDynamicResizeConf, SyncDynamicResizeOp);

}  // namespace oneflow
