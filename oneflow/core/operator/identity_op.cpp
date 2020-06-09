#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

template<typename T>
class IdentityOpTpl final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdentityOpTpl);
  IdentityOpTpl() = default;
  ~IdentityOpTpl() override = default;

  void InitFromOpConf() override {
    EnrollInputBn("in");
    EnrollOutputBn("out")->set_const_inplace_ibn("in");
  }
  const PbMessage& GetCustomizedConf() const override { return T::GetCustomizedConf(op_conf()); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    const auto bns = StdVec2PbRpf<std::string>({"in", "out"});
    SbpSignatureBuilder().PartialSum(bns).Build(sbp_sig_list->mutable_sbp_signature()->Add());
    const int64_t num_axes = JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes();
    SbpSignatureBuilder().Split(bns, 0).MakeSplitSignatureListBuilder(num_axes).Build(sbp_sig_list);
    return Maybe<void>::Ok();
  }
};

struct IdentityOp {
  static const PbMessage& GetCustomizedConf(const OperatorConf& op_conf) {
    return op_conf.identity_conf();
  }
};
REGISTER_OP(OperatorConf::kIdentityConf, IdentityOpTpl<IdentityOp>);

struct CopyOp {
  static const PbMessage& GetCustomizedConf(const OperatorConf& op_conf) {
    return op_conf.copy_conf();
  }
};
REGISTER_OP(OperatorConf::kCopyConf, IdentityOpTpl<CopyOp>);

}  // namespace oneflow
