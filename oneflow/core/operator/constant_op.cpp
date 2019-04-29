#include "oneflow/core/operator/constant_op.h"

namespace oneflow {

namespace {

class ConstantOpSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantOpSbpSignatureRule);
  ~ConstantOpSbpSignatureRule() override = default;

  ConstantOpSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": S(0) -> B"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    if (op().op_conf().constant_conf().has_tick()) {
      CHECK(SbpInferHint4Ibn("tick").is_data_split());
      (*bn2sbp)["tick"].mutable_split_parallel()->set_axis(0);
    }
    (*bn2sbp)["out"].mutable_broadcast_parallel();
  }
};

}  // namespace

void ConstantOp::InitFromOpConf() {
  CHECK(op_conf().has_constant_conf());
  if (op_conf().constant_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

const PbMessage& ConstantOp::GetCustomizedConf() const { return op_conf().constant_conf(); }

void ConstantOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                int64_t record_piece_size) const {
  CHECK_EQ(parallel_ctx->policy(), ParallelPolicy::kDataParallel);
  const ConstantOpConf& conf = op_conf().constant_conf();
  const DataType& data_type =
      conf.has_data_type() ? conf.data_type() : Global<JobDesc>::Get()->DefaultDataType();
  std::vector<int64_t> dim_vec;
  if (conf.use_device_piece_size_as_dim0()) {
    CHECK_EQ(record_piece_size % parallel_ctx->parallel_num(), 0);
    dim_vec.push_back(record_piece_size / parallel_ctx->parallel_num());
  }
  if (conf.has_shape()) {
    dim_vec.insert(dim_vec.end(), conf.shape().dim().cbegin(), conf.shape().dim().cend());
  }
  if (dim_vec.empty()) { dim_vec.push_back(1); }
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->set_data_type(data_type);
  out->mut_shape() = Shape(dim_vec);
}

void ConstantOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new ConstantOpSbpSignatureRule(this));
}

void ConstantOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->mutable_constant_conf()->set_random_seed(NewRandomSeed());
  const DataType& data_type = GetBlobDesc4BnInOp("out")->data_type();
  if (op_conf().constant_conf().has_initializer()) {
    *kernel_conf->mutable_constant_conf()->mutable_initializer() =
        op_conf().constant_conf().initializer();
  } else if (IsFloatingDataType(data_type)) {
    InitializerConf conf;
    conf.mutable_constant_conf()->set_value(0);
    *kernel_conf->mutable_constant_conf()->mutable_initializer() = conf;
  } else if (IsIntegralDataType(data_type)) {
    InitializerConf conf;
    conf.mutable_constant_int_conf()->set_value(0);
    *kernel_conf->mutable_constant_conf()->mutable_initializer() = conf;
  } else {
    UNIMPLEMENTED();
  }
  kernel_conf->set_data_type(data_type);
}

void ConstantOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = false;
}

REGISTER_OP(OperatorConf::kConstantConf, ConstantOp);

}  // namespace oneflow
