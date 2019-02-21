#ifndef ONEFLOW_CORE_OPERATOR_ROI_ALIGN_OP_H_
#define ONEFLOW_CORE_OPERATOR_ROI_ALIGN_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RoIAlignOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoIAlignOp);
  RoIAlignOp() = default;
  ~RoIAlignOp() = default;

  const PbMessage& GetCustomizedConf() const override;

  void InitFromOpConf() override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ROI_ALIGN_OP_H_
