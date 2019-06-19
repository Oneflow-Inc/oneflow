#ifndef ONEFLOW_CORE_OPERATOR_RING_BOXING_OP_H_
#define ONEFLOW_CORE_OPERATOR_RING_BOXING_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RingBoxingOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RingBoxingOp);
  RingBoxingOp() = default;
  ~RingBoxingOp() override = default;

  void InitFromOpConf() override;

 protected:
  virtual const RingBoxingConf& GetCustomizedBoxingConf() const;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

class RingReduceScatterOp final : public RingBoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RingReduceScatterOp);
  RingReduceScatterOp() = default;
  ~RingReduceScatterOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

class RingAllGatherOp final : public RingBoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RingAllGatherOp);
  RingAllGatherOp() = default;
  ~RingAllGatherOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

class RingAllReduceOp final : public RingBoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RingAllReduceOp);
  RingAllReduceOp() = default;
  ~RingAllReduceOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RING_BOXING_OP_H_
