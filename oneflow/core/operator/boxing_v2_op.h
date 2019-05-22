#ifndef ONEFLOW_CORE_OPERATOR_BOXING_V2_OP_H_
#define ONEFLOW_CORE_OPERATOR_BOXING_V2_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BoxingV2Op : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingV2Op);
  BoxingV2Op() = default;
  ~BoxingV2Op() override = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 protected:
  virtual const BoxingV2Conf& GetCustomizedBoxingConf() const;
  virtual void VirtualInferBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {}
  virtual void VirtualInitFromOpConf(){};

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

class BoxingV2CopyOp final : public BoxingV2Op {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingV2CopyOp);
  BoxingV2CopyOp() = default;
  ~BoxingV2CopyOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
};

class BoxingV2AddOp final : public BoxingV2Op {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingV2AddOp);
  BoxingV2AddOp() = default;
  ~BoxingV2AddOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  void VirtualInitFromOpConf() override;
  void VirtualInferBlobDescs(const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BOXING_V2_OP_H_
