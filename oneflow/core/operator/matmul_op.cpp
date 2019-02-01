#include "oneflow/core/operator/matmul_op.h"
#include "oneflow/core/common/balanced_splitter.h"
namespace oneflow {

namespace {

std::unique_ptr<const OpParallelSignature> MakeMatmulOpParallelSignature_DMS_MS_2_P(
    const MatmulOp* op) {
  std::string desc = op->op_name() + ": (S, S) -> P";
  auto GetMatchResult =
      [op](const std::function<const LogicalBlobParallelDesc&(const std::string&)>&,
           const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
           const ParallelContext* parallel_ctx) {
        const auto& b_lbpd_hint = LbpdHint4BnInOp("b");
        if (!b_lbpd_hint.has_model_split()) { return MakeOpParallelMatchSignatureMismatch(); }
        int32_t b_expected_split_axis = (op->op_conf().matmul_conf().transpose_b() ? 1 : 0);
        if (b_lbpd_hint.model_split().axis() != b_expected_split_axis) {
          return MakeOpParallelMatchSignatureMismatch();
        }
        if (parallel_ctx->policy() == kModelParallel) { return MakeOpParallelMatchSuccess(); }
        return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kModelParallel);
      };
  auto GenSignature = [op](
                          const std::function<const LbpdHint&(const std::string&)>& LbpdHint4BnInOp,
                          HashMap<std::string, LogicalBlobParallelDesc>* signature) {
    int32_t a_split_axis = (op->op_conf().matmul_conf().transpose_a() ? 0 : 1);
    const auto& b_lbpd_hint = LbpdHint4BnInOp("b");
    (*signature)["a"].mutable_split_parallel()->set_axis(a_split_axis);
    (*signature)["b"].mutable_split_parallel()->set_axis(b_lbpd_hint.model_split().axis());
    (*signature)["out"].mutable_partial_sum_parallel();
  };
  return std::make_unique<OpParallelSignature>(desc, GetMatchResult, GenSignature);
}

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

void MatmulOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(MakeDataSplitOpParallelSignature(this));
  op_parallel_signatures->emplace_back(MakeOpParallelSignature_DS_MC_2_DS(this));
  auto IsValidSplit = [this](int32_t axis) {
    int32_t b_expected_split_axis = (op_conf().matmul_conf().transpose_b() ? 0 : 1);
    return axis == b_expected_split_axis;
  };
  op_parallel_signatures->emplace_back(MakeOpParallelSignature_DC_MS_2_MS(this, IsValidSplit));
  op_parallel_signatures->emplace_back(MakeMatmulOpParallelSignature_DMS_MS_2_P(this));
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

void MatmulOp::InferOutputBlobLbpdHint(
    std::function<LbpdHint*(const std::string&)> LbpdHint4BnInOp,
    std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
    const ParallelContext* parallel_context) const {
  CHECK_EQ(ShapeNumAxes4BnInOp("a"), ShapeNumAxes4BnInOp("b"));
  const auto& b_lbpd_hint = *LbpdHint4BnInOp("b");
  if (ShapeNumAxes4BnInOp("b") == 2 && b_lbpd_hint.has_model_split()) {
    int32_t b_model_split_axis = b_lbpd_hint.model_split().axis();
    if (op_conf().matmul_conf().transpose_b()) {
      if (b_model_split_axis == 0) {
        LbpdHint4BnInOp("out")->mutable_data_split()->set_axis(1);
      } else if (b_model_split_axis == 1) {
        LbpdHint4BnInOp("out")->mutable_data_partial_sum();
      } else {
        UNIMPLEMENTED();
      }
    } else {
      if (b_model_split_axis == 0) {
        LbpdHint4BnInOp("out")->mutable_data_partial_sum();
      } else if (b_model_split_axis == 1) {
        LbpdHint4BnInOp("out")->mutable_data_split()->set_axis(1);
      } else {
        UNIMPLEMENTED();
      }
    }
  } else {
    CHECK_GT(ShapeNumAxes4BnInOp("b"), 2);
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    LbpdHint4BnInOp("out")->mutable_data_split()->set_axis(0);
  }
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
