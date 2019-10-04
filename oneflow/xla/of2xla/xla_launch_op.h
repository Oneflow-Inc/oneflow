#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_LAUNCH_OP_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_LAUNCH_OP_H_

#include <unordered_map>
#include "oneflow/xla/of2xla/xla_graph.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class XlaLaunchOp : public Operator {
 public:
  void InitFromOpConf() override;
  
  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

  LogicalNode* NewProperLogicalNode() const override {
    if (is_model_update_) {
      return new OptimizerLogicalNode;
    }
    return new NormalForwardLogicalNode;
  }

 private:
  Maybe<void> InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp)
    const override;

  typedef std::function<Maybe<const SbpInferHint*>(const std::string&)>
      SbpInferHint4IbnFunc;
  Maybe<void> InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    SbpInferHint4IbnFunc SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const override;

  bool is_model_update_ = false;
  std::shared_ptr<mola::XlaLaunchGraph> subgraph_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_LAUNCH_OP_H_
