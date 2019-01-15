#ifndef ONEFLOW_CORE_OPERATOR_SCALAR_ADD_OP_H_
#define ONEFLOW_CORE_OPERATOR_SCALAR_ADD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ScalarAddOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarAddOp);
  ScalarAddOp() = default;
  ~ScalarAddOp() = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  bool IsElemWiseOp() const override { return true; }
  const PbMessage& GetCustomizedConf() const override { return op_conf().scalar_add_conf(); }
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }

 private:
  void InferOutBlobModelSplitAxis(std::function<int64_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int64_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override {
    NaiveInferOutBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SCALAR_ADD_OP_H_
