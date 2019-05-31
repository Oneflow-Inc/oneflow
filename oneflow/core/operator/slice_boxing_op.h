#ifndef ONEFLOW_CORE_OPERATOR_SLICE_BOXING_OP_H_
#define ONEFLOW_CORE_OPERATOR_SLICE_BOXING_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SliceBoxingOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingOp);
  SliceBoxingOp() = default;
  ~SliceBoxingOp() override = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 protected:
  virtual const SliceBoxingConf& GetCustomizedBoxingConf() const;
  virtual void VirtualInferBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {}
  virtual void VirtualInitFromOpConf(){};

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

class SliceBoxingCopyOp final : public SliceBoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingCopyOp);
  SliceBoxingCopyOp() = default;
  ~SliceBoxingCopyOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
};

class SliceBoxingAddOp final : public SliceBoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingAddOp);
  SliceBoxingAddOp() = default;
  ~SliceBoxingAddOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  void VirtualInitFromOpConf() override;
  void VirtualInferBlobDescs(const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SLICE_BOXING_OP_H_
