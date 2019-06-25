#ifndef ONEFLOW_CORE_OPERATOR_LEVEL_MAP_OP_H_
#define ONEFLOW_CORE_OPERATOR_LEVEL_MAP_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LevelMapOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LevelMapOp);
  LevelMapOp() = default;
  ~LevelMapOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*) const override;
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LEVEL_MAP_OP_H_
