#include "oneflow/core/operator/operator.h"

namespace oneflow {

class WhereOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WhereOp);
  WhereOp() = default;
  ~WhereOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_where_conf());
    EnrollInputBn("condition", false);
    EnrollInputBn("x");
    EnrollInputBn("y");
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override { return this->op_conf().where_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* condition_blob_desc = GetBlobDesc4BnInOp("condition");
    const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
    const BlobDesc* y_blob_desc = GetBlobDesc4BnInOp("y");
    CHECK_EQ_OR_RETURN(condition_blob_desc->shape(), x_blob_desc->shape());
    CHECK_EQ_OR_RETURN(condition_blob_desc->shape(), y_blob_desc->shape());
    CHECK_EQ_OR_RETURN(x_blob_desc->data_type(), y_blob_desc->data_type());
    *GetBlobDesc4BnInOp("out") = *x_blob_desc;
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    WhereKernelConf* conf = kernel_conf->mutable_where_conf();
    conf->set_cond_type(GetBlobDesc4BnInOp("condition")->data_type());
    conf->set_value_type(GetBlobDesc4BnInOp("x")->data_type());
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .MakeSplitSignatureListBuilder(
            JUST(LogicalBlobDesc4Ibn(input_bns().Get(0)))->shape().NumAxes())
        .Build(sbp_sig_list);
    SbpSignatureBuilder()
        .Broadcast("condition")
        .PartialSum("x")
        .PartialSum("y")
        .PartialSum("out")
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kWhereConf, WhereOp);

}  // namespace oneflow
