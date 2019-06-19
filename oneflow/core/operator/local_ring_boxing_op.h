#ifndef ONEFLOW_CORE_OPERATOR_LOCAL_RING_BOXING_OP_H_
#define ONEFLOW_CORE_OPERATOR_LOCAL_RING_BOXING_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LocalRingBoxingOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalRingBoxingOp);
  LocalRingBoxingOp() = default;
  ~LocalRingBoxingOp() override = default;

  void InitFromOpConf() override;

 protected:
  virtual const LocalRingBoxingConf& GetCustomizedBoxingConf() const;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

class LocalRingAllReduceOp final : public LocalRingBoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalRingAllReduceOp);
  LocalRingAllReduceOp() = default;
  ~LocalRingAllReduceOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_LOCAL_RING_BOXING_OP_H_
