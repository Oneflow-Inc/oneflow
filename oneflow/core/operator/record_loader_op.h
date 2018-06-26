#ifndef ONEFLOW_CORE_OPERATOR_RECORD_LOADER_OP_H_
#define ONEFLOW_CORE_OPERATOR_RECORD_LOADER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RecordLoaderOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordLoaderOp);
  RecordLoaderOp() = default;
  ~RecordLoaderOp() = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RECORD_LOADER_OP_H_
