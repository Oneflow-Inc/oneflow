#ifndef ONEFLOW_CORE_OPERATOR_SLICE_OP_H_
#define ONEFLOW_CORE_OPERATOR_SLICE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SliceOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceOp);
  SliceOp() = default;
  ~SliceOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SLICE_OP_H_
