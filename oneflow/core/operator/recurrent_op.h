#ifndef ONEFLOW_CORE_OPERATOR_RECURRENT_OP_H_
#define ONEFLOW_CORE_OPERATOR_RECURRENT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RecurrentOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecurrentOp);
  RecurrentOp() = default;
  virtual ~RecurrentOp() = default;

  void InitFromOpConf() override;
  bool IsRecurrentOp() const override { return true; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  int32_t ModelSplitAxis() const override { return 1; }
  int32_t MaxModelSplitNum() const override;

 private:
  virtual void VirtualInitFromOpConf() { UNIMPLEMENTED(); }
  virtual void VirtualInferBlobDescs(std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                                     const ParallelContext* parallel_ctx) const {
    UNIMPLEMENTED();
  }
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RECURRENT_OP_H_
