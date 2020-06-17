#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/job/mirrored_sig_infer_hint.h"

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

class MirroredCastOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MirroredCastOp);
  MirroredCastOp() = default;
  virtual ~MirroredCastOp() override = default;

  void InitFromOpConf() override {
    EnrollInputBn("in");
    EnrollOutputBn("out")->set_const_inplace_ibn("in");
  }
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
};

namespace {

class CastToMirroredOp : public MirroredCastOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastToMirroredOp);
  CastToMirroredOp() = default;
  virtual ~CastToMirroredOp() override = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().cast_to_mirrored_conf(); }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    CHECK_NE_OR_RETURN(op_conf().cast_to_mirrored_conf().sbp_parallel().parallel_type_case(),
                       SbpParallel::PARALLEL_TYPE_NOT_SET)
        << "attribute sbp_parallel not set.";
    auto* sbp_signature = sbp_sig_list->mutable_sbp_signature()->Add();
    auto* map = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*map)["in"] = op_conf().cast_to_mirrored_conf().sbp_parallel();
    (*map)["out"] = op_conf().cast_to_mirrored_conf().sbp_parallel();
    return Maybe<void>::Ok();
  }
  Maybe<void> InferMirroredSignature(
      std::function<Maybe<const MirroredSigInferHint*>(const std::string&)>
          MirroredSigInferHint4Ibn,
      bool is_mirrored_parallel_view_conf, const ParallelDesc& parallel_desc) override {
    const auto& in_infer_hint = *JUST(MirroredSigInferHint4Ibn("in"));
    CHECK_OR_RETURN(!in_infer_hint.is_mirrored_parallel_view())
        << "error use of CastToMirroredOp. `in' shouldn't be a mirrored blob";
    CHECK_EQ_OR_RETURN(in_infer_hint.parallel_desc().parallel_num(), parallel_desc.parallel_num());
    MutOptMirroredParallel("in")->clear_mirrored_parallel();
    MutOptMirroredParallel("out")->mutable_mirrored_parallel();
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kCastToMirroredConf, CastToMirroredOp);

}  // namespace

namespace {

class CastFromMirroredOp : public MirroredCastOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastFromMirroredOp);
  CastFromMirroredOp() = default;
  virtual ~CastFromMirroredOp() override = default;

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().cast_from_mirrored_conf();
  }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    CHECK_NE_OR_RETURN(op_conf().cast_from_mirrored_conf().sbp_parallel().parallel_type_case(),
                       SbpParallel::PARALLEL_TYPE_NOT_SET)
        << "attribute sbp_parallel not set.";
    auto* sbp_signature = sbp_sig_list->mutable_sbp_signature()->Add();
    auto* map = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*map)["in"] = op_conf().cast_from_mirrored_conf().sbp_parallel();
    (*map)["out"] = op_conf().cast_from_mirrored_conf().sbp_parallel();
    return Maybe<void>::Ok();
  }
  Maybe<void> InferMirroredSignature(
      std::function<Maybe<const MirroredSigInferHint*>(const std::string&)>
          MirroredSigInferHint4Ibn,
      bool is_mirrored_parallel_view_conf, const ParallelDesc& parallel_desc) override {
    const auto& in_infer_hint = *JUST(MirroredSigInferHint4Ibn("in"));
    CHECK_OR_RETURN(in_infer_hint.is_mirrored_parallel_view())
        << "error use of CastFromMirroredOp. `in' should be a mirrored blob";
    CHECK_EQ_OR_RETURN(in_infer_hint.parallel_desc().parallel_num(), parallel_desc.parallel_num());
    MutOptMirroredParallel("in")->mutable_mirrored_parallel();
    MutOptMirroredParallel("out")->clear_mirrored_parallel();
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kCastFromMirroredConf, CastFromMirroredOp);

}  // namespace

}  // namespace oneflow
