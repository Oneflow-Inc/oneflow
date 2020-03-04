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
    *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kSyncDynamicResizeConf, SyncDynamicResizeOp);

}  // namespace oneflow
