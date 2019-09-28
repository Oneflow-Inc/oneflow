#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

class NcclHierarchicalOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclHierarchicalOp);
  NcclHierarchicalOp() = default;
  ~NcclHierarchicalOp() override = default;

  void InitFromOpConf() override {
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false);
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    *GetBlobDesc4BnInOp(SoleObn()) = *GetBlobDesc4BnInOp(SoleIbn());
    return Maybe<void>::Ok();
  }

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override {
    LogicalBlobId ret;
    ret.set_op_name(op_name());
    ret.set_blob_name("out");
    return ret;
  }
};

class NcclHierarchicalReduceOp final : public NcclHierarchicalOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclHierarchicalReduceOp);
  NcclHierarchicalReduceOp() = default;
  ~NcclHierarchicalReduceOp() override = default;

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().nccl_hierarchical_reduce_conf();
  }
};

class NcclHierarchicalAllReduceOp final : public NcclHierarchicalOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclHierarchicalAllReduceOp);
  NcclHierarchicalAllReduceOp() = default;
  ~NcclHierarchicalAllReduceOp() override = default;

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().nccl_hierarchical_all_reduce_conf();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    kernel_conf->mutable_nccl_hierarchical_all_reduce_conf()->set_need_do_all_reduce(
        parallel_ctx->parallel_id()
            % op_conf().nccl_hierarchical_all_reduce_conf().device_num_of_each_node()
        == 0);
  }
};

class NcclHierarchicalBroadcastOp final : public NcclHierarchicalOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclHierarchicalBroadcastOp);
  NcclHierarchicalBroadcastOp() = default;
  ~NcclHierarchicalBroadcastOp() override = default;

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().nccl_hierarchical_broadcast_conf();
  }
};

REGISTER_OP(OperatorConf::kNcclHierarchicalReduceConf, NcclHierarchicalReduceOp);
REGISTER_OP(OperatorConf::kNcclHierarchicalAllReduceConf, NcclHierarchicalAllReduceOp);
REGISTER_OP(OperatorConf::kNcclHierarchicalBroadcastConf, NcclHierarchicalBroadcastOp);
}  // namespace oneflow
