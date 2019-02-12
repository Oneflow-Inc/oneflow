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

std::unique_ptr<const OpParallelSignature> MakeGatherOpParallelSignature_DC_MS_2_P(
    const GatherOp* op) {
  std::string desc = op->op_name() + ": (C, S) -> P";
  auto GetMatchResult =
      [op](const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
           const ParallelContext* parallel_ctx) {
        const SbpInferHint& in_sbp_infer_hint = SbpInferHint4BnInOp("in");
        if (!in_sbp_infer_hint.is_model_split()) { return MakeOpParallelMatchSignatureMismatch(); }
        if (in_sbp_infer_hint.model_split().axis() != 0) {
          return MakeOpParallelMatchSignatureMismatch();
        }
        if (parallel_ctx->policy() == kModelParallel) { return MakeOpParallelMatchSuccess(); }
        return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kModelParallel);
      };
  auto GenSignature =
      [op](const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
           HashMap<std::string, SbpParallel>* signature) {
        (*signature)["indices"].mutable_broadcast_parallel();
        (*signature)["in"].mutable_split_parallel()->set_axis(0);
        (*signature)["out"].mutable_partial_sum_parallel();
      };
  return std::make_unique<OpParallelSignature>(desc, GetMatchResult, GenSignature);
}

}  // namespace

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
  std::vector<int64_t> dim_vec;
  dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin(),
                 in->shape().dim_vec().cbegin() + axis);
  dim_vec.insert(dim_vec.end(), indices->shape().dim_vec().cbegin(),
                 indices->shape().dim_vec().cend());
  dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin() + axis + 1,
                 in->shape().dim_vec().end());
  out->mut_shape() = Shape(dim_vec);
}

void GatherOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const int64_t axis = GetGatherAxis(op_conf().gather_conf(), GetBlobDesc4BnInOp("in"));
  kernel_conf->mutable_gather_conf()->set_axis(axis);
}

void GatherOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(MakeDataSplitOpParallelSignature(this));
  op_parallel_signatures->emplace_back(MakeOpParallelSignature_DS_MC_2_DS(this));
  auto GtZero = [](int32_t axis) { return axis > 0; };
  op_parallel_signatures->emplace_back(MakeOpParallelSignature_DC_MS_2_MS(this, GtZero));
  op_parallel_signatures->emplace_back(MakeGatherOpParallelSignature_DC_MS_2_P(this));
}

void GatherOp::InferOutputBlobSbpInferHint(
    std::function<SbpInferHint*(const std::string&)> SbpInferHint4BnInOp,
    const ParallelContext* parallel_context) const {
  const SbpInferHint& indices_sbp_infer_hint = *SbpInferHint4BnInOp("indices");
  CHECK(indices_sbp_infer_hint.is_data_blob());
  const SbpInferHint& in_sbp_infer_hint = *SbpInferHint4BnInOp("in");
  const int64_t in_num_axes = in_sbp_infer_hint.num_axes();
  const int64_t gather_axis = GetGatherAxis(op_conf().gather_conf(), in_num_axes);
  if (in_sbp_infer_hint.is_model_split()) {
    if (in_sbp_infer_hint.model_split().axis() == 0) {
      SbpInferHint4BnInOp("out")->mutable_data_partial_sum();
    } else {
      CHECK_GT(in_sbp_infer_hint.model_split().axis(), gather_axis);
      CHECK_LT(in_sbp_infer_hint.model_split().axis(), in_num_axes);
      int32_t axis = in_sbp_infer_hint.model_split().axis() + indices_sbp_infer_hint.num_axes() - 1;
      SbpInferHint4BnInOp("out")->mutable_data_split()->set_axis(axis);
    }
  } else {
    CHECK(in_sbp_infer_hint.is_model_broadcast() || in_sbp_infer_hint.is_data_split()
          || in_sbp_infer_hint.is_data_partial_sum());
    SbpInferHint4BnInOp("out")->mutable_data_split()->set_axis(0);
  }
}

REGISTER_OP(OperatorConf::kGatherConf, GatherOp);

}  // namespace oneflow
