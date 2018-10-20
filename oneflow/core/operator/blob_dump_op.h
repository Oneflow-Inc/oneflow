#ifndef ONEFLOW_CORE_OPERATOR_BLOB_DUMP_OP_H_
#define ONEFLOW_CORE_OPERATOR_BLOB_DUMP_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class BlobDumpOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobDumpOp);
  BlobDumpOp() = default;
  ~BlobDumpOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  virtual LogicalNode* NewProperLogicalNode() { return new PrintLogicalNode; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override {}

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId pibn2lbi(const std::string& pb_input_bn) const override;
  LogicalBlobId Lbi4InputBn(const std::string& input_bn) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BLOB_DUMP_OP_H_
