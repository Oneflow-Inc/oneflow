#include "oneflow/core/operator/gather_op.h"

namespace oneflow {

namespace {

int64_t GetGatherAxis(const GatherOpConf& conf, int64_t num_axes) {
  const int64_t axis = conf.axis() < 0 ? num_axes + conf.axis() : conf.axis();
  CHECK_GE(axis, 0);
  CHECK_LT(axis, num_axes);
  return axis;
}

int64_t GetGatherAxis(const GatherOpConf& conf, const BlobDesc* in_blob_desc) {
  return GetGatherAxis(conf, in_blob_desc->shape().NumAxes());
}

class GatherDataParallelSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherDataParallelSbpSignatureRule);
  ~GatherDataParallelSbpSignatureRule() override = default;

  explicit GatherDataParallelSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (S, B) -> S"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const SbpInferHint& in_sbp_infer_hint = SbpInferHint4BnInOp("in");
    if (!in_sbp_infer_hint.sbp_parallel().has_broadcast_parallel()) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    const SbpInferHint& indices_sbp_infer_hint = SbpInferHint4BnInOp("indices");
    if (!indices_sbp_infer_hint.sbp_parallel().has_split_parallel()) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    if (parallel_desc.policy() == kDataParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kDataParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    const int64_t gather_axis =
        GetGatherAxis(op().op_conf().gather_conf(), SbpInferHint4BnInOp("in").num_axes());
    const int64_t indices_split_axis =
        SbpInferHint4BnInOp("indices").sbp_parallel().split_parallel().axis();
    (*bn2sbp)["indices"].mutable_split_parallel()->set_axis(indices_split_axis);
    (*bn2sbp)["in"].mutable_broadcast_parallel();
    (*bn2sbp)["out"].mutable_split_parallel()->set_axis(gather_axis + indices_split_axis);
  }
};

class GatherModelParallelSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherModelParallelSbpSignatureRule);
  ~GatherModelParallelSbpSignatureRule() override = default;

  explicit GatherModelParallelSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (B, S) -> S"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const SbpInferHint& in_sbp_infer_hint = SbpInferHint4BnInOp("in");
    if (!in_sbp_infer_hint.sbp_parallel().has_split_parallel()) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    const int64_t gather_axis =
        GetGatherAxis(op().op_conf().gather_conf(), SbpInferHint4BnInOp("in").num_axes());
    if (in_sbp_infer_hint.sbp_parallel().split_parallel().axis() == gather_axis) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    const SbpInferHint& indices_sbp_infer_hint = SbpInferHint4BnInOp("indices");
    if (!indices_sbp_infer_hint.sbp_parallel().has_broadcast_parallel()) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    if (parallel_desc.policy() == kModelParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kModelParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    const int64_t gather_axis =
        GetGatherAxis(op().op_conf().gather_conf(), SbpInferHint4BnInOp("in").num_axes());
    const int64_t in_split_axis = SbpInferHint4BnInOp("in").sbp_parallel().split_parallel().axis();
    (*bn2sbp)["indices"].mutable_broadcast_parallel();
    (*bn2sbp)["in"].mutable_split_parallel()->set_axis(in_split_axis);
    (*bn2sbp)["out"].mutable_split_parallel()->set_axis(
        in_split_axis < gather_axis
            ? in_split_axis
            : in_split_axis + SbpInferHint4BnInOp("indices").num_axes() - 1);
  }
};

}  // namespace

Shape GatherGetOutShape(const Shape& in, const Shape& indices, const int64_t axis) {
  std::vector<int64_t> dim_vec;
  dim_vec.insert(dim_vec.end(), in.dim_vec().cbegin(), in.dim_vec().cbegin() + axis);
  dim_vec.insert(dim_vec.end(), indices.dim_vec().cbegin(), indices.dim_vec().cend());
  dim_vec.insert(dim_vec.end(), in.dim_vec().cbegin() + axis + 1, in.dim_vec().end());
  return Shape(dim_vec);
}

void GatherOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_conf());
  EnrollInputBn("indices", false);
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& GatherOp::GetCustomizedConf() const { return op_conf().gather_conf(); }

bool GatherOp::IsInputBlobAllowedModelSplit(const std::string& ibn) const {
  CHECK(std::find(input_bns().begin(), input_bns().end(), ibn) != input_bns().end());
  return ibn == "in";
}

void GatherOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK(IsIntegralDataType(indices->data_type()));
  CHECK_GT(indices->shape().NumAxes(), 0);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GT(in->shape().NumAxes(), 0);
  const int64_t axis = GetGatherAxis(op_conf().gather_conf(), in);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->mut_shape() = Shape(GatherGetOutShape(in->shape(), indices->shape(), axis));
}

void GatherOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const int64_t axis = GetGatherAxis(op_conf().gather_conf(), GetBlobDesc4BnInOp("in"));
  kernel_conf->mutable_gather_conf()->set_axis(axis);
}

void GatherOp::GetSbpSignatureRules(
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new GatherDataParallelSbpSignatureRule(this));
  rules->emplace_back(new GatherModelParallelSbpSignatureRule(this));
}

int32_t GatherOp::OutputBlobModelSplitAxis(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    const std::string& obn) const {
  UNIMPLEMENTED();
}

REGISTER_OP(OperatorConf::kGatherConf, GatherOp);

}  // namespace oneflow
