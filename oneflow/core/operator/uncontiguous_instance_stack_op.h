#ifndef ONEFLOW_CORE_OPERATOR_UNCONTIGUOUS_INSTANCE_STACK_OP_H_
#define ONEFLOW_CORE_OPERATOR_UNCONTIGUOUS_INSTANCE_STACK_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class UncontiguousInstanceStackOp final : public Operator {
 public:
  OF_DISALLOW_COPY(UncontiguousInstanceStackOp);
  UncontiguousInstanceStackOp() = default;
  ~UncontiguousInstanceStackOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().uncontiguous_instance_stack_conf();
  }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_UNCONTIGUOUS_INSTANCE_STACK_OP_H_
