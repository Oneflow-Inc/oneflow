#ifndef ONEFLOW_CORE_OPERATOR_EMBEDDING_LOOKUP_OP_H_
#define ONEFLOW_CORE_OPERATOR_EMBEDDING_LOOKUP_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class EmbeddingLookupOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingLookupOp);
  EmbeddingLookupOp() = default;
  ~EmbeddingLookupOp() = default;

  void InitFromOpConf() override;
  bool IsEmbeddingLookupOp() const override { return true; }
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_EMBEDDING_LOOKUP_OP_H_
