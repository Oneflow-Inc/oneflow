#ifndef ONEFLOW_CORE_OPERATOR_SHARED_MODEL_DIFF_ADD_OP_H_
#define ONEFLOW_CORE_OPERATOR_SHARED_MODEL_DIFF_ADD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SharedModelDiffAddOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SharedModelDiffAddOp);
  SharedModelDiffAddOp() = default;
  ~SharedModelDiffAddOp() = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  void InferOutBlobModelSplitAxis(std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }

  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override { return GenPackedLbi(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SHARED_MODEL_DIFF_ADD_OP_H_
