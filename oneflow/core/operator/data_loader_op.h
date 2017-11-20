#ifndef ONEFLOW_CORE_OPERATOR_DATA_LOADER_OP_H_
#define ONEFLOW_CORE_OPERATOR_DATA_LOADER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class DataLoaderOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoaderOp);
  DataLoaderOp() = default;
  ~DataLoaderOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;
  bool IsDataLoaderOp() const override { return true; }

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DATA_LOADER_OP_H_
