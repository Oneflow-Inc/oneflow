#include "oneflow/core/operator/gather_grad_op.h"
#include "oneflow/core/operator/gather_op.h"

namespace oneflow {

namespace {

class GatherGradDataParallelSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherGradDataParallelSbpSignatureRule);
  ~GatherGradDataParallelSbpSignatureRule() override = default;

  explicit GatherGradDataParallelSbpSignatureRule(const Operator* op)
      : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": S -> P"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const SbpParallel& indices_sbp_parallel = SbpInferHint4BnInOp("indices").sbp_parallel();
    if (!indices_sbp_parallel.has_split_parallel()) { return MakeSbpSigMatchSignatureMismatch(); }
    const SbpParallel& out_diff_sbp_parallel = SbpInferHint4BnInOp("out_diff").sbp_parallel();
    if (!out_diff_sbp_parallel.has_split_parallel()) { return MakeSbpSigMatchSignatureMismatch(); }
    if (out_diff_sbp_parallel.split_parallel().axis()
        != indices_sbp_parallel.split_parallel().axis()
               + op().op_conf().gather_grad_conf().axis()) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    if (parallel_desc.policy() == kDataParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kDataParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    const int64_t indices_split_axis =
        SbpInferHint4BnInOp("indices").sbp_parallel().split_parallel().axis();
    (*bn2sbp)["indices"].mutable_split_parallel()->set_axis(indices_split_axis);
    (*bn2sbp)["out_diff"].mutable_split_parallel()->set_axis(
        indices_split_axis + op().op_conf().gather_grad_conf().axis());
    (*bn2sbp)["in_diff"].mutable_partial_sum_parallel();
  }
};

class GatherGradModelParallelSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherGradModelParallelSbpSignatureRule);
  ~GatherGradModelParallelSbpSignatureRule() override = default;

  explicit GatherGradModelParallelSbpSignatureRule(const Operator* op)
      : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (B, S) -> S"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const SbpParallel& indices_sbp_parallel = SbpInferHint4BnInOp("indices").sbp_parallel();
    if (!indices_sbp_parallel.has_broadcast_parallel()) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    const SbpParallel& out_diff_sbp_parallel = SbpInferHint4BnInOp("out_diff").sbp_parallel();
    if (!out_diff_sbp_parallel.has_split_parallel()) { return MakeSbpSigMatchSignatureMismatch(); }
    const int64_t gather_axis = op().op_conf().gather_grad_conf().axis();
    if (out_diff_sbp_parallel.split_parallel().axis() >= gather_axis
        && out_diff_sbp_parallel.split_parallel().axis()
               < gather_axis + SbpInferHint4BnInOp("indices").num_axes()) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    if (parallel_desc.policy() == kModelParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kModelParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    const int64_t out_diff_split_axis =
        SbpInferHint4BnInOp("out_diff").sbp_parallel().split_parallel().axis();
    (*bn2sbp)["indices"].mutable_broadcast_parallel();
    (*bn2sbp)["out_diff"].mutable_split_parallel()->set_axis(out_diff_split_axis);
    const int64_t gather_axis = op().op_conf().gather_grad_conf().axis();
    const int64_t in_diff_split_axis =
        (out_diff_split_axis < gather_axis)
            ? out_diff_split_axis
            : out_diff_split_axis - SbpInferHint4BnInOp("indices").num_axes() + 1;
    CHECK_NE(gather_axis, in_diff_split_axis);
    (*bn2sbp)["in_diff"].mutable_split_parallel()->set_axis(in_diff_split_axis);
  }
};

}  // namespace

void GatherGradOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_grad_conf());
  EnrollInputBn("indices", false);
  EnrollInputBn("out_diff", false);
  EnrollOutputBn("in_diff", false);
}

const PbMessage& GatherGradOp::GetCustomizedConf() const { return op_conf().gather_grad_conf(); }

bool GatherGradOp::IsInputBlobAllowedModelSplit(const std::string& ibn) const { return false; }

void GatherGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  const GatherGradOpConf& conf = op_conf().gather_grad_conf();
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK(IsIntegralDataType(indices->data_type()));
  const BlobDesc* out_diff = GetBlobDesc4BnInOp("out_diff");
  std::vector<int64_t> in_diff_dim_vec;
  in_diff_dim_vec.insert(in_diff_dim_vec.end(), out_diff->shape().dim_vec().cbegin(),
                         out_diff->shape().dim_vec().cbegin() + conf.axis());
  in_diff_dim_vec.push_back(conf.gather_dim_size());
  in_diff_dim_vec.insert(
      in_diff_dim_vec.end(),
      out_diff->shape().dim_vec().cbegin() + conf.axis() + indices->shape().NumAxes(),
      out_diff->shape().dim_vec().end());
  BlobDesc* in_diff = GetBlobDesc4BnInOp("in_diff");
  in_diff->set_data_type(out_diff->data_type());
  in_diff->mut_shape() = Shape(in_diff_dim_vec);
}

void GatherGradOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new GatherGradDataParallelSbpSignatureRule(this));
  rules->emplace_back(new GatherGradModelParallelSbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kGatherGradConf, GatherGradOp);

}  // namespace oneflow
