#include "oneflow/core/operator/gather_grad_op.h"
#include "oneflow/core/operator/gather_op.h"

namespace oneflow {

namespace {

class GatherGradDataParallelOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherGradDataParallelOpParallelSignature);
  ~GatherGradDataParallelOpParallelSignature() override = default;

  explicit GatherGradDataParallelOpParallelSignature(const Operator* op)
      : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": S -> P"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.policy() == kDataParallel) { return MakeOpParallelMatchSuccess(); }
    return MakeOpParallelMatchParallelPolicyError(parallel_desc.policy(), kDataParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["indices"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["out_diff"].mutable_split_parallel()->set_axis(0);
    (*bn2sbp)["in_diff"].mutable_partial_sum_parallel();
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

void GatherGradOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  // TODO: support model parallel
  op_parallel_signatures->emplace_back(new GatherGradDataParallelOpParallelSignature(this));
}

REGISTER_OP(OperatorConf::kGatherGradConf, GatherGradOp);

}  // namespace oneflow
