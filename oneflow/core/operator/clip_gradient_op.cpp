#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ClipGradientOp : public Operator {
 public:
  ClipGradientOp() = default;
  virtual ~ClipGradientOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 private:
  typedef std::function<bool*(const std::string&)> HasBatchDim4BnInOpFunc;
  void InferHasBatchDim(
      HasBatchDim4BnInOpFunc HasBatchDim4BnInOp) const override {
    *HasBatchDim4BnInOp("out") = *HasBatchDim4BnInOp("gradient");
  }

  typedef std::function<const BlobDesc&(const std::string&)>
      LogicalBlobDesc4IbnFunc;
  void GetSbpSignatures(const LogicalBlobDesc4IbnFunc& LogicalBlobDesc4Ibn,
                        SbpSignatureList* sbp_sig_list) const override;
};

void ClipGradientOp::InitFromOpConf() {
  CHECK(op_conf().has_clip_gradient_conf());
  EnrollInputBn("gradient", false);
  EnrollInputBn("instance_num_diff", false);
  EnrollOutputBn("out", false)->set_mutable_inplace_ibn("gradient");
}

const PbMessage& ClipGradientOp::GetCustomizedConf() const {
  CHECK(op_conf().has_clip_gradient_conf());
  return op_conf().clip_gradient_conf();
}

void ClipGradientOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("gradient");
}

typedef std::function<const BlobDesc&(const std::string&)>
      LogicalBlobDesc4IbnFunc;
void ClipGradientOp::GetSbpSignatures(
    const LogicalBlobDesc4IbnFunc& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const Shape &gradient_shape = LogicalBlobDesc4Ibn("gradient").shape();
  for (int i = 0; i < gradient_shape.NumAxes(); ++i) {
    SbpSignatureBuilder()
        .Split("gradient", i)
        .Split("out", i)
        .Broadcast("instance_num_diff")
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
}

REGISTER_OP(OperatorConf::kClipGradientConf, ClipGradientOp);

}  // namespace oneflow
