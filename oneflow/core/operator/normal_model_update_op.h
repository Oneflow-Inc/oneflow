#ifndef ONEFLOW_CORE_OPERATOR_NORMAL_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_NORMAL_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NormalModelUpdtOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalModelUpdtOp);
  virtual ~NormalModelUpdtOp() = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override;

 protected:
  NormalModelUpdtOp() = default;
  virtual void MdUpdtVirtualInitFromOpConf() {}
  virtual void MdUpdtVirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*) const {}

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { UNIMPLEMENTED(); }
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }

  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NORMAL_MODEL_UPDATE_OP_H_
