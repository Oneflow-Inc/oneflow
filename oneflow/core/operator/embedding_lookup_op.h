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
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  int32_t ModelSplitAxis() const override { return 1; }
  int32_t MaxModelSplitNum() const override { return op_conf().embedding_lookup_conf().units(); }

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    NaiveInferOutputBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp,
                                       parallel_context);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_EMBEDDING_LOOKUP_OP_H_
