#ifndef ONEFLOW_CORE_OPERATOR_RECORD_LOAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_RECORD_LOAD_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class DataLoadOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoadOp);
  DataLoadOp() = default;
  ~DataLoadOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx,
                      int64_t record_piece_size) const override;
  // void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)>
  // GetBlobDesc4BnInOp,
  //                           const ParallelContext*, KernelConf*) const override;

  LogicalNode* NewProperLogicalNode() override { return new RecordLoadLogicalNode; }

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RECORD_LOAD_OP_H_
