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
    EnrollInputBn("lhs");
    EnrollInputBn("rhs");
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override { return this->op_conf().where_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    // input: condition
    const BlobDesc* condition = GetBlobDesc4BnInOp("condition");
    CHECK(IsIntegralDataType(condition->data_type()));
    // input: lhs and rhs
    const BlobDesc* lhs = GetBlobDesc4BnInOp("lhs");
    const BlobDesc* rhs = GetBlobDesc4BnInOp("rhs");
    const Shape shape = condition->shape();
    CHECK_EQ_OR_RETURN(shape, lhs->shape());
    CHECK_EQ_OR_RETURN(shape, rhs->shape());
    CHECK_EQ_OR_RETURN(lhs->data_type(), rhs->data_type());
    // output
    *GetBlobDesc4BnInOp("out") = *lhs;

    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    WhereKernelConf* conf = kernel_conf->mutable_where_conf();
    conf->set_cond_type(GetBlobDesc4BnInOp("condition")->data_type());
    conf->set_value_type(GetBlobDesc4BnInOp("lhs")->data_type());
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
        .Split("condition", 0)
        .Split("lhs", 0)
        .Split("rhs", 0)
        .Split("out", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kWhereConf, WhereOp);

}  // namespace oneflow
