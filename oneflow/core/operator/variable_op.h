#ifndef ONEFLOW_CORE_OPERATOR_VARIABLE_OP_H_
#define ONEFLOW_CORE_OPERATOR_VARIABLE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class VariableOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(VariableOp);
  VariableOp()
      : Operator(),
        is_fw_inplace_(std::make_unique<bool>(false)),
        is_bw_inplace_(std::make_unique<bool>(false)) {}
  ~VariableOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  int32_t ModelSplitAxis() const override;

  void set_is_fw_inplace(bool val) const { *is_fw_inplace_ = val; }
  void set_is_bw_inplace(bool val) const { *is_bw_inplace_ = val; }

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void GetOpParallelSignatures(
      std::vector<std::unique_ptr<const OpParallelSignature>>*) const override;
  void InferOutputBlobSbpInferHint(
      std::function<SbpInferHint*(const std::string&)> SbpInferHint4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override;
  void InferIsModelBlob4OutputBlobs(
      std::function<bool*(const std::string&)> IsModelBlob4BnInOp) const;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*) const override;

  std::unique_ptr<bool> is_fw_inplace_;
  std::unique_ptr<bool> is_bw_inplace_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_VARIABLE_OP_H_
