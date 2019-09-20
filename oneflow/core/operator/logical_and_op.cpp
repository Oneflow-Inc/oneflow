#include "oneflow/core/operator/operator.h"

namespace oneflow {
void LogicalAndOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {}

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
    CHECK_EQ(lhs->shape(), rhs->shape());
    CHECK_EQ(lhs->has_dim0_valid_num_field(), rhs->has_dim0_valid_num_field());
    CHECK_EQ(lhs->has_instance_shape_field(), rhs->has_instance_shape_field());
    // output
    *GetBlobDesc4BnInOp("out") = *lhs;
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kLogicalAndConf, LogicalAndOp);

}  // namespace oneflow
