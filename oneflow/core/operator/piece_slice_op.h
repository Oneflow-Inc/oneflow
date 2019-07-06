#ifndef ONEFLOW_CORE_OPERATOR_PIECE_SLICE_OP_H_
#define ONEFLOW_CORE_OPERATOR_PIECE_SLICE_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class PieceSliceOp final : public Operator {
 public:
  OF_DISALLOW_COPY(PieceSliceOp);
  PieceSliceOp() = default;
  ~PieceSliceOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().piece_slice_conf(); }
  LogicalNode* NewProperLogicalNode() { return new PieceSliceForwardLogicalNode; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferOutputBlobTimeShape(std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
                                const ParallelContext* parallel_ctx,
                                Shape* time_shape) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PIECE_SLICE_OP_H_
