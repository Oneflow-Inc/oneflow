#include "oneflow/core/operator/matmul_op.h"
#include "oneflow/core/common/balanced_splitter.h"
namespace oneflow {

namespace {

class Matmul_MS_MS_2_P_SbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Matmul_MS_MS_2_P_SbpSignatureRule);
  ~Matmul_MS_MS_2_P_SbpSignatureRule() override = default;

  Matmul_MS_MS_2_P_SbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (S, S) -> P"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    const auto& b_sbp_infer_hint = SbpInferHint4Ibn("b");
    if (!b_sbp_infer_hint.is_model_split()) { return MakeSbpSigMatchSignatureMismatch(); }
    int32_t b_expected_split_axis = (op().op_conf().matmul_conf().transpose_b() ? 1 : 0);
    if (b_sbp_infer_hint.split_axis() != b_expected_split_axis) {
      return MakeSbpSigMatchSignatureMismatch();
    }
    if (parallel_desc.policy() == kModelParallel) { return MakeSbpSigMatchSuccess(); }
    return MakeSbpSigMatchParallelPolicyError(parallel_desc.policy(), kModelParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    int32_t a_split_axis = (op().op_conf().matmul_conf().transpose_a() ? 0 : 1);
    (*bn2sbp)["a"].mutable_split_parallel()->set_axis(a_split_axis);
    (*bn2sbp)["b"] = SbpInferHint4Ibn("b").sbp_parallel();
    (*bn2sbp)["out"].mutable_partial_sum_parallel();
  }
};

}  // namespace

void MatmulOp::InitFromOpConf() {
  CHECK(op_conf().has_matmul_conf());
  EnrollInputBn("a");
  EnrollInputBn("b");
  EnrollOutputBn("out");
  EnrollFwBufBn("fw_buf");
  EnrollBwBufBn("bw_buf");
}

const PbMessage& MatmulOp::GetCustomizedConf() const { return op_conf().matmul_conf(); }

bool MatmulOp::IsInputBlobAllowedModelSplit(const std::string& ibn) const {
  CHECK(std::find(input_bns().begin(), input_bns().end(), ibn) != input_bns().end());
  return ibn == "b";
}

void MatmulOp::GetSbpSignatureRules(
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(MakeDataSplitSbpSignatureRule(this));
  rules->emplace_back(Make_DS_MB_2_DS_SbpSignatureRule(this));
  auto IsValidSplit = [this](int32_t axis) {
    int32_t b_expected_split_axis = (op_conf().matmul_conf().transpose_b() ? 0 : 1);
    return axis == b_expected_split_axis;
  };
  rules->emplace_back(Make_DB_MS_2_MS_SbpSignatureRule(this, IsValidSplit));
  rules->emplace_back(new Matmul_MS_MS_2_P_SbpSignatureRule(this));
}

void MatmulOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const MatmulOpConf& conf = op_conf().matmul_conf();
  BlobDesc* a_blob_desc = GetBlobDesc4BnInOp("a");
  BlobDesc* b_blob_desc = GetBlobDesc4BnInOp("b");
  CHECK_EQ(a_blob_desc->shape().NumAxes(), b_blob_desc->shape().NumAxes());
  CHECK_GE(a_blob_desc->shape().NumAxes(), 2);
  size_t num_axes = a_blob_desc->shape().NumAxes();
  if (conf.transpose_a()) {
    CHECK(!a_blob_desc->has_dim0_valid_num_field());
    CHECK(!a_blob_desc->has_dim1_valid_num_field());
    CHECK(!a_blob_desc->has_dim2_valid_num_field());
  }
  if (conf.transpose_b()) {
    CHECK(!b_blob_desc->has_dim0_valid_num_field());
    CHECK(!b_blob_desc->has_dim1_valid_num_field());
    CHECK(!b_blob_desc->has_dim2_valid_num_field());
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *a_blob_desc;
  FOR_RANGE(int32_t, i, 0, num_axes - 2) {
    CHECK_EQ(a_blob_desc->shape().At(i), b_blob_desc->shape().At(i));
  }
  int64_t a_dim_index = conf.transpose_a() ? num_axes - 1 : num_axes - 2;
  out_blob_desc->mut_shape().Set(num_axes - 2, a_blob_desc->shape().At(a_dim_index));
  int64_t b_dim_index = conf.transpose_b() ? num_axes - 2 : num_axes - 1;
  out_blob_desc->mut_shape().Set(num_axes - 1, b_blob_desc->shape().At(b_dim_index));
  int64_t a_mid_dim_index = conf.transpose_a() ? num_axes - 2 : num_axes - 1;
  int64_t b_mid_dim_index = conf.transpose_b() ? num_axes - 1 : num_axes - 2;
  CHECK_EQ(a_blob_desc->shape().At(a_mid_dim_index), b_blob_desc->shape().At(b_mid_dim_index));
  if (device_type() == DeviceType::kGPU && num_axes >= 3) {
    int batch_num = a_blob_desc->shape().Count(0, num_axes - 2);
    // Assume gpu address is 64 bit
    BlobDesc* fw_buf_blob_desc = GetBlobDesc4BnInOp("fw_buf");
    *fw_buf_blob_desc = *out_blob_desc;
    fw_buf_blob_desc->mut_shape() = {3 * batch_num};
    fw_buf_blob_desc->set_data_type(DataType::kInt64);
    fw_buf_blob_desc->set_has_data_id_field(false);
  }
}

int32_t MatmulOp::OutputBlobModelSplitAxis(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    const std::string& obn) const {
  CHECK_EQ(SbpInferHint4Ibn("a").num_axes(), SbpInferHint4Ibn("b").num_axes());
  const auto& b_sbp_infer_hint = SbpInferHint4Ibn("b");
  CHECK_EQ(SbpInferHint4Ibn("b").num_axes(), 2);
  CHECK(b_sbp_infer_hint.is_model_split());
  int32_t b_model_split_axis = b_sbp_infer_hint.split_axis();
  if (op_conf().matmul_conf().transpose_b()) {
    if (b_model_split_axis == 0) { return 1; }
  } else {
    if (b_model_split_axis == 1) { return 1; }
  }
  UNIMPLEMENTED();
  return -1;
}

void MatmulOp::InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext*) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  size_t num_axes = out_blob_desc->shape().NumAxes();
  if (device_type() == DeviceType::kGPU && num_axes >= 3) {
    BlobDesc* bw_buf_blob_desc = GetBlobDesc4BnInOp("bw_buf");
    int32_t batch_num = out_blob_desc->shape().Count(0, num_axes - 2);
    *bw_buf_blob_desc = *out_blob_desc;
    bw_buf_blob_desc->mut_shape() = {3 * batch_num};
    bw_buf_blob_desc->set_data_type(DataType::kInt64);
    bw_buf_blob_desc->set_has_data_id_field(false);
  }
}

REGISTER_OP(OperatorConf::kMatmulConf, MatmulOp);

}  // namespace oneflow
