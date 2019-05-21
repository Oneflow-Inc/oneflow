#ifndef ONEFLOW_CORE_OPERATOR_BOXING_COPY_OP_H_
#define ONEFLOW_CORE_OPERATOR_BOXING_COPY_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BoxingCopyOpBase : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingCopyOpBase);
  BoxingCopyOpBase() = default;
  ~BoxingCopyOpBase() override = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

class BoxingCopyOp final : public BoxingCopyOpBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingCopyOp);
  BoxingCopyOp() = default;
  ~BoxingCopyOp() override = default;

  const PbMessage& GetCustomizedConf() const override;
};

class BoxingCopyAddOp final : public BoxingCopyOpBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingCopyAddOp);
  BoxingCopyAddOp() = default;
  ~BoxingCopyAddOp() override = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BOXING_COPY_OP_H_
