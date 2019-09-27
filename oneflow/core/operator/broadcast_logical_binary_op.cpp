#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastLogicalBinaryOp : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastLogicalBinaryOp);
  BroadcastLogicalBinaryOp() = default;
  virtual ~BroadcastLogicalBinaryOp() = default;

 private:
  Maybe<void> VirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp) const override {
    GetBlobDesc4BnInOp("out")->set_data_type(DataType::kInt8);
    return Maybe<void>::Ok();
  }
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx,
      std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const override {
    kernel_conf->set_data_type(GetBlobDesc4BnInOp("a")->data_type());
  }
};

#define DEFINE_BROADCAST_OP_CLASS(camel_case, snake_case)                   \
  class Broadcast##camel_case##Op final : public BroadcastLogicalBinaryOp { \
   public:                                                                  \
    OF_DISALLOW_COPY_AND_MOVE(Broadcast##camel_case##Op);                   \
    Broadcast##camel_case##Op() = default;                                  \
    ~Broadcast##camel_case##Op() = default;                                 \
    const PbMessage& GetCustomizedConf() const override {                   \
      return op_conf().broadcast_##snake_case##_conf();                     \
    }                                                                       \
  };                                                                        \
  REGISTER_OP(OperatorConf::kBroadcast##camel_case##Conf, Broadcast##camel_case##Op)

DEFINE_BROADCAST_OP_CLASS(Equal, equal);
DEFINE_BROADCAST_OP_CLASS(NotEqual, not_equal);
DEFINE_BROADCAST_OP_CLASS(GreaterThan, greater_than);
DEFINE_BROADCAST_OP_CLASS(GreaterEqual, greater_equal);
DEFINE_BROADCAST_OP_CLASS(LessThan, less_than);
DEFINE_BROADCAST_OP_CLASS(LessEqual, less_equal);
DEFINE_BROADCAST_OP_CLASS(LogicalAnd, logical_and);
}  // namespace oneflow
