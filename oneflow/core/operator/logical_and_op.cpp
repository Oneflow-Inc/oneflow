#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LogicalAndOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogicalAndOp);
  LogicalAndOp() = default;
  ~LogicalAndOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_logical_and_conf());
    EnrollInputBn("lhs", false);
    EnrollInputBn("rhs", false);
    EnrollOutputBn("out", false);
  }

  const PbMessage& GetCustomizedConf() const override { return this->op_conf().logical_and_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    // input: lhs and rhs
    const BlobDesc* lhs = GetBlobDesc4BnInOp("lhs");
    const BlobDesc* rhs = GetBlobDesc4BnInOp("rhs");
    CHECK_EQ_OR_RETURN(lhs->shape(), rhs->shape());
    // output
    *GetBlobDesc4BnInOp("out") = *lhs;
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kLogicalAndConf, LogicalAndOp);

}  // namespace oneflow
