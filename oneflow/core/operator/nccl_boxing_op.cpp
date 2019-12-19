#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NcclBoxingOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclBoxingOp);
  NcclBoxingOp() = default;
  ~NcclBoxingOp() override = default;

  void InitFromOpConf() override {
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false);
  }

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override {
    return GetMsgFromCustomizedConf<const LogicalBlobId&>("lbi");
  }

  LogicalBlobId obn2lbi(const std::string& output_bn) const override {
    return GetMsgFromCustomizedConf<const LogicalBlobId&>("lbi");
  }
};

class NcclBoxingReduceScatterOp : public NcclBoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclBoxingReduceScatterOp);
  NcclBoxingReduceScatterOp() = default;
  ~NcclBoxingReduceScatterOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().nccl_boxing_reduce_scatter_conf();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    CHECK_GT_OR_RETURN(in->shape().NumAxes(), 0);
    CHECK_EQ_OR_RETURN(in->shape().At(0) % parallel_ctx->parallel_num(), 0);
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    out->mut_shape().Set(0, in->shape().At(0) / parallel_ctx->parallel_num());
    return Maybe<void>::Ok();
  }
};

class NcclBoxingAllGatherOp : public NcclBoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclBoxingAllGatherOp);
  NcclBoxingAllGatherOp() = default;
  ~NcclBoxingAllGatherOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().nccl_boxing_all_gather_conf();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    CHECK_GT_OR_RETURN(in->shape().NumAxes(), 0);
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    out->mut_shape().Set(0, in->shape().At(0) * parallel_ctx->parallel_num());
    return Maybe<void>::Ok();
  }
};

class NcclBoxingAllReduceOp : public NcclBoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclBoxingAllReduceOp);
  NcclBoxingAllReduceOp() = default;
  ~NcclBoxingAllReduceOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().nccl_boxing_all_reduce_conf();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    CHECK_GT_OR_RETURN(in->shape().NumAxes(), 0);
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kNcclBoxingReduceScatterConf, NcclBoxingReduceScatterOp);
REGISTER_OP(OperatorConf::kNcclBoxingAllGatherConf, NcclBoxingAllGatherOp);
REGISTER_OP(OperatorConf::kNcclBoxingAllReduceConf, NcclBoxingAllReduceOp);

}  // namespace oneflow
