#ifndef ONEFLOW_CORE_OPERATOR_NORMAL_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_NORMAL_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NormalModelUpdtOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalModelUpdtOp);
  virtual ~NormalModelUpdtOp() = default;

  void InitFromOpConf() override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  virtual const PbMessage& GetCustomizedConf() const override;

 protected:
  NormalModelUpdtOp() = default;
  virtual void MdUpdtVirtualInitFromOpConf() {}
  virtual Maybe<void> MdUpdtVirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*) const {
    return Maybe<void>::Ok();
  }

  virtual const HashSet<std::string> AlwaysBroadcastParallelBns() const = 0;

 private:
  Maybe<void> InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;

  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NORMAL_MODEL_UPDATE_OP_H_
