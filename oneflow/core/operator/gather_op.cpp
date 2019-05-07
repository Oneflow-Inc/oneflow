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

class Gather_DB_MS_2_P_OpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Gather_DB_MS_2_P_OpParallelSignature);
  ~Gather_DB_MS_2_P_OpParallelSignature() override = default;

  Gather_DB_MS_2_P_OpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": (C, S) -> P"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelContext* parallel_ctx) const override {
    const SbpInferHint& in_sbp_infer_hint = SbpInferHint4BnInOp("in");
    if (!in_sbp_infer_hint.is_model_split()) { return MakeOpParallelMatchSignatureMismatch(); }
    if (in_sbp_infer_hint.split_axis() != 0) { return MakeOpParallelMatchSignatureMismatch(); }
    if (parallel_ctx->policy() == kModelParallel) { return MakeOpParallelMatchSuccess(); }
    return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kModelParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["indices"].mutable_broadcast_parallel();
    (*bn2sbp)["in"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["out"].mutable_partial_sum_parallel();
  }
};

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
  auto dim_vec = in->shape().dim_vec();
  auto insert_dim_vec = indices->shape().dim_vec();
  auto insert_pos = dim_vec.erase(dim_vec.begin() + axis);
  dim_vec.insert(insert_pos, insert_dim_vec.begin(), insert_dim_vec.end());
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
  op_parallel_signatures->emplace_back(Make_DS_MB_2_DS_OpParallelSignature(this));
  auto GtZero = [](int32_t axis) { return axis > 0; };
  op_parallel_signatures->emplace_back(Make_DB_MS_2_MS_OpParallelSignature(this, GtZero));
  op_parallel_signatures->emplace_back(new Gather_DB_MS_2_P_OpParallelSignature(this));
}

int32_t GatherOp::OutputBlobModelSplitAxis(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    const std::string& obn) const {
  const SbpInferHint& indices_sbp_infer_hint = SbpInferHint4Ibn("indices");
  CHECK(indices_sbp_infer_hint.is_data_blob());
  const SbpInferHint& in_sbp_infer_hint = SbpInferHint4Ibn("in");
  const int64_t in_num_axes = in_sbp_infer_hint.num_axes();
  const int64_t gather_axis = GetGatherAxis(op_conf().gather_conf(), in_num_axes);
  CHECK(in_sbp_infer_hint.is_model_split());
  CHECK_GT(in_sbp_infer_hint.split_axis(), 0);
  CHECK_GT(in_sbp_infer_hint.split_axis(), gather_axis);
  CHECK_LT(in_sbp_infer_hint.split_axis(), in_num_axes);
  return in_sbp_infer_hint.split_axis() + indices_sbp_infer_hint.num_axes() - 1;
}

REGISTER_OP(OperatorConf::kGatherConf, GatherOp);

}  // namespace oneflow
