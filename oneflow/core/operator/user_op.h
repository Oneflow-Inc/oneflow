#ifndef ONEFLOW_CORE_OPERATOR_USER_OP_H_
#define ONEFLOW_CORE_OPERATOR_USER_OP_H_

#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class UserOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserOp);
  UserOp() = default;
  ~UserOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().user_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override;
  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext*, const SbpSignature* sbp_signature,
                                std::function<void(OpContext*)> EnrollOpCtx) const override;

  Symbol<OperatorConf> GetOpConfWithoutOpNameAndLbn() const override;

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override;
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
  Maybe<void> InferBatchAxis(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const override;
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx,
      std::function<const BlobDesc&(const std::string&)> LogicalBlobDesc4BnInOp) const override;

  const user_op::OpRegistryResult* val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_USER_OP_H_
