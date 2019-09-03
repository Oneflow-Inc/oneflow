#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AdamOptimizerOp : public Operator {
 public:
  AdamOptimizerOp() = default;
  virtual ~AdamOptimizerOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("weight");
    *GetBlobDesc4BnInOp("out_m") = *GetBlobDesc4BnInOp("m");
    *GetBlobDesc4BnInOp("out_v") = *GetBlobDesc4BnInOp("v");
  };

 private:
  typedef std::function<bool*(const std::string&)> HasBatchDim4BnInOpFunc;
  void InferHasBatchDim(
      HasBatchDim4BnInOpFunc HasBatchDim4BnInOp) const override {
    for (const auto& ibn : input_bns()) {
      CHECK_EQ(*HasBatchDim4BnInOp(ibn), false);
      *HasBatchDim4BnInOp("out") = false;
      *HasBatchDim4BnInOp("out_m") = false;
      *HasBatchDim4BnInOp("out_v") = false;
    }
  }

  typedef std::function<const BlobDesc&(const std::string&)>
      LogicalBlobDesc4IbnFunc;
  void GetSbpSignatures(const LogicalBlobDesc4IbnFunc& LogicalBlobDesc4Ibn,
                        SbpSignatureList* sbp_sig_list) const override;
};

void AdamOptimizerOp::InitFromOpConf() {
  CHECK(op_conf().has_adam_optimizer_conf());
  std::vector<std::string> input_bns{"gradient", "instance_num_diff",
                                     "learning_rate"};
  for (const std::string &bn : input_bns) {
    EnrollInputBn(bn, false);
  }
  EnrollInputBn("m")->set_is_mutable(true);
  EnrollInputBn("v")->set_is_mutable(true);
  EnrollInputBn("weight")->set_is_mutable(true);
  EnrollOutputBn("out")->set_mutable_inplace_ibn("weight");
  EnrollOutputBn("out_m")->set_mutable_inplace_ibn("m");
  EnrollOutputBn("out_v")->set_mutable_inplace_ibn("v");
}

const PbMessage& AdamOptimizerOp::GetCustomizedConf() const {
  CHECK(op_conf().has_adam_optimizer_conf());
  return op_conf().adam_optimizer_conf();
}

typedef std::function<const BlobDesc&(const std::string&)>
      LogicalBlobDesc4IbnFunc;
void AdamOptimizerOp::GetSbpSignatures(
    const LogicalBlobDesc4IbnFunc& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const Shape &weight_shape = LogicalBlobDesc4Ibn("weight").shape();
  for (int i = 0; i < weight_shape.NumAxes(); ++i) {
    SbpSignatureBuilder()
        .Split({"out", "out_m", "out_v", "gradient", "weight", "m", "v"}, i)
        .Broadcast({"instance_num_diff", "learning_rate"})
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
}

REGISTER_OP(OperatorConf::kAdamOptimizerConf, AdamOptimizerOp);

}  // namespace oneflow
