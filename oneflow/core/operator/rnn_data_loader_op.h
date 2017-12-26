#ifndef ONEFLOW_CORE_OPERATOR_RNN_DATA_LOADER_OP_H_
#define ONEFLOW_CORE_OPERATOR_RNN_DATA_LOADER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RnnDataLoaderOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RnnDataLoaderOp);
  RnnDataLoaderOp() = default;
  ~RnnDataLoaderOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;
  bool IsRnnDataLoderOp() const { return true; }

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RNN_DATA_LOADER_OP_H_
