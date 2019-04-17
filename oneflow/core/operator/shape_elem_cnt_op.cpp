#include "oneflow/core/operator/shape_elem_cnt_op.h"

namespace oneflow {

namespace {

HashSet<int32_t> GetInclusiveAxes(const ShapeElemCntOpConf& conf, int32_t num_axes) {
  HashSet<int32_t> ret;
  if (conf.has_exclude_axis_conf()) {
    HashSet<int32_t> exclude_axes(conf.exclude_axis_conf().axis().begin(),
                                  conf.exclude_axis_conf().axis().end());
    FOR_RANGE(int32_t, i, 0, num_axes) {
      if (exclude_axes.find(i) == exclude_axes.end()
          && exclude_axes.find(i - num_axes) == exclude_axes.end()) {
        ret.insert(i);
      }
    }
  } else if (conf.has_include_axis_conf()) {
    for (int32_t axis : conf.include_axis_conf().axis()) {
      if (axis < 0) { axis += num_axes; }
      CHECK_GE(axis, 0);
      CHECK_LT(axis, num_axes);
      ret.insert(axis);
    }
  } else if (conf.has_range_axis_conf()) {
    TODO();
  } else {
    UNIMPLEMENTED();
  }
  return ret;
}

class ShapeElemCntOpSplitSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ShapeElemCntOpSplitSbpSignatureRule);
  ~ShapeElemCntOpSplitSbpSignatureRule() override = default;

  explicit ShapeElemCntOpSplitSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (S,) -> (P,)"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const SbpInferHint& in_sbp_infer_hint = SbpInferHint4BnInOp("x");
    if (!in_sbp_infer_hint.sbp_parallel().has_split_parallel()) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    const int32_t num_axes = SbpInferHint4BnInOp("x").num_axes();
    const int32_t split_axis = SbpInferHint4BnInOp("x").sbp_parallel().split_parallel().axis();
    const HashSet<int32_t>& inclusive_axis =
        GetInclusiveAxes(op().op_conf().shape_elem_cnt_conf(), num_axes);
    if (inclusive_axis.find(split_axis) != inclusive_axis.end()) {
      return MakeSbpSigMatchSuccess();
    } else {
      return MakeSbpSigMatchSignatureMismatch();
    }
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*bn2sbp)["x"] = SbpInferHint4BnInOp("x").sbp_parallel();
    (*bn2sbp)["y"].mutable_partial_sum_parallel();
  }
};

class ShapeElemCntOpBroadcastSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ShapeElemCntOpBroadcastSbpSignatureRule);
  ~ShapeElemCntOpBroadcastSbpSignatureRule() override = default;

  explicit ShapeElemCntOpBroadcastSbpSignatureRule(const Operator* op)
      : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (S,) -> (B,)"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    const SbpInferHint& in_sbp_infer_hint = SbpInferHint4BnInOp("x");
    if (!in_sbp_infer_hint.sbp_parallel().has_split_parallel()) { return MakeSbpSigMatchSuccess(); }
    const int32_t num_axes = SbpInferHint4BnInOp("x").num_axes();
    const int32_t split_axis = SbpInferHint4BnInOp("x").sbp_parallel().split_parallel().axis();
    const HashSet<int32_t>& inclusive_axis =
        GetInclusiveAxes(op().op_conf().shape_elem_cnt_conf(), num_axes);
    if (inclusive_axis.find(split_axis) != inclusive_axis.end()) {
      return MakeSbpSigMatchSignatureMismatch();
    } else {
      return MakeSbpSigMatchSuccess();
    }
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*bn2sbp)["x"] = SbpInferHint4BnInOp("x").sbp_parallel();
    (*bn2sbp)["y"].mutable_broadcast_parallel();
  }
};

}  // namespace

void ShapeElemCntOp::InitFromOpConf() {
  EnrollInputBn("x", false)->set_use_header_only(true);
  EnrollOutputBn("y", false);
}

const PbMessage& ShapeElemCntOp::GetCustomizedConf() const {
  return op_conf().shape_elem_cnt_conf();
}

void ShapeElemCntOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  GetBlobDesc4BnInOp("y")->set_data_type(DataType::kInt32);
  GetBlobDesc4BnInOp("y")->mut_shape() = Shape({1});
}

void ShapeElemCntOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(MakeSoleIbnBroadcastSbpSignatureRule(this));
  rules->emplace_back(new ShapeElemCntOpSplitSbpSignatureRule(this));
  rules->emplace_back(new ShapeElemCntOpBroadcastSbpSignatureRule(this));
}

void ShapeElemCntOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  int32_t num_axes = GetBlobDesc4BnInOp("x")->shape().NumAxes();
  const HashSet<int32_t>& inclusive_axis =
      GetInclusiveAxes(op_conf().shape_elem_cnt_conf(), num_axes);
  *kernel_conf->mutable_shape_elem_cnt_conf()->mutable_axis() = {inclusive_axis.begin(),
                                                                 inclusive_axis.end()};
}

REGISTER_OP(OperatorConf::kShapeElemCntConf, ShapeElemCntOp);

}  // namespace oneflow
