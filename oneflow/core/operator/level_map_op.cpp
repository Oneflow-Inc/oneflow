#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LevelMapOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LevelMapOp);
  LevelMapOp() = default;
  ~LevelMapOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_level_map_conf());
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false);
  }
  const PbMessage& GetCustomizedConf() const override { return this->op_conf().level_map_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    // input: boxes
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    CHECK_EQ_OR_RETURN(in->shape().dim_vec().back(), 4);
    // output: levels
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    out->mut_shape() =
        Shape(DimVector(in->shape().dim_vec().begin(), in->shape().dim_vec().end() - 1));
    out->set_data_type(DataType::kInt32);

    return Maybe<void>::Ok();
  }

 private:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf* kernel_conf) const override {
    kernel_conf->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
  }
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder().Split("in", 0).Split("out", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kLevelMapConf, LevelMapOp);

}  // namespace oneflow
