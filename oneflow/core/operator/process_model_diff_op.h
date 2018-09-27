#ifndef ONEFLOW_CORE_OPERATOR_PROCESS_MODEL_DIFF_OP_H_
#define ONEFLOW_CORE_OPERATOR_PROCESS_MODEL_DIFF_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ProcessModelDiffOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ProcessModelDiffOp);
  ProcessModelDiffOp() = default;
  ~ProcessModelDiffOp() = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override { return GenPackedLbi(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PROCESS_MODEL_DIFF_OP_H_
