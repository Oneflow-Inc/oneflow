#ifndef ONEFLOW_CORE_OPERATOR_DEVICE_TICK_OP_H_
#define ONEFLOW_CORE_OPERATOR_DEVICE_TICK_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class DeviceTickOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceTickOp);
  DeviceTickOp() = default;
  ~DeviceTickOp() = default;

  void InitFromOpConf() override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().device_tick_conf(); }
  LogicalNode* NewProperLogicalNode() const override { return new DeviceTickLogicalNode; }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DEVICE_TICK_OP_H_
