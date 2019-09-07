#include "oneflow/core/operator/matmul_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void MatmulOp::InitFromOpConf() {
  CHECK(op_conf().has_matmul_conf());
  EnrollInputBn("a");
  EnrollInputBn("b");
  EnrollOutputBn("out");
  EnrollTmpBn("fw_buf");
}

const PbMessage& MatmulOp::GetCustomizedConf() const { return op_conf().matmul_conf(); }

Maybe<void> MatmulOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const MatmulOpConf& conf = op_conf().matmul_conf();
  BlobDesc* a_blob_desc = GetBlobDesc4BnInOp("a");
  BlobDesc* b_blob_desc = GetBlobDesc4BnInOp("b");
  CHECK_EQ_OR_RETURN(a_blob_desc->shape().NumAxes(), b_blob_desc->shape().NumAxes());
  CHECK_GE_OR_RETURN(a_blob_desc->shape().NumAxes(), 2);
  size_t num_axes = a_blob_desc->shape().NumAxes();
  if (conf.transpose_a()) {
    CHECK_OR_RETURN(!a_blob_desc->has_dim0_valid_num_field());
    CHECK_OR_RETURN(!a_blob_desc->has_dim1_valid_num_field());
    CHECK_OR_RETURN(!a_blob_desc->has_dim2_valid_num_field());
  }
  if (conf.transpose_b()) {
    CHECK_OR_RETURN(!b_blob_desc->has_dim0_valid_num_field());
    CHECK_OR_RETURN(!b_blob_desc->has_dim1_valid_num_field());
    CHECK_OR_RETURN(!b_blob_desc->has_dim2_valid_num_field());
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *a_blob_desc;
  FOR_RANGE(int32_t, i, 0, num_axes - 2) {
    CHECK_EQ_OR_RETURN(a_blob_desc->shape().At(i), b_blob_desc->shape().At(i));
  }
  int64_t a_dim_index = conf.transpose_a() ? num_axes - 1 : num_axes - 2;
  out_blob_desc->mut_shape().Set(num_axes - 2, a_blob_desc->shape().At(a_dim_index));
  int64_t b_dim_index = conf.transpose_b() ? num_axes - 2 : num_axes - 1;
  out_blob_desc->mut_shape().Set(num_axes - 1, b_blob_desc->shape().At(b_dim_index));
  int64_t a_mid_dim_index = conf.transpose_a() ? num_axes - 2 : num_axes - 1;
  int64_t b_mid_dim_index = conf.transpose_b() ? num_axes - 1 : num_axes - 2;
  CHECK_EQ_OR_RETURN(a_blob_desc->shape().At(a_mid_dim_index),
                     b_blob_desc->shape().At(b_mid_dim_index));
  if (device_type() == DeviceType::kGPU && num_axes >= 3) {
    int batch_num = a_blob_desc->shape().Count(0, num_axes - 2);
    // Assume gpu address is 64 bit
    BlobDesc* fw_buf_blob_desc = GetBlobDesc4BnInOp("fw_buf");
    *fw_buf_blob_desc = *out_blob_desc;
    fw_buf_blob_desc->mut_shape() = {3 * batch_num};
    fw_buf_blob_desc->set_data_type(DataType::kInt64);
    fw_buf_blob_desc->set_has_data_id_field(false);
  }
  return Maybe<void>::Ok();
}

Maybe<void> MatmulOp::InferBatchAxis(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  const MatmulOpConf& conf = op_conf().matmul_conf();
  int32_t num_axes = LogicalBlobDesc4Ibn("a").shape().NumAxes();
  if (num_axes > 2) {
    if (BatchAxis4BnInOp("a")->has_value()) {
      *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("a");
    } else if (BatchAxis4BnInOp("b")->has_value()) {
      *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("b");
    } else {
      BatchAxis4BnInOp("out")->clear_value();
    }
  } else if (num_axes == 2) {
    OptInt64 a_batch_axis(*BatchAxis4BnInOp("a"));
    if (a_batch_axis.has_value() && conf.transpose_a()) {
      a_batch_axis.set_value(1 - a_batch_axis.value());
    }
    OptInt64 b_batch_axis(*BatchAxis4BnInOp("b"));
    if (b_batch_axis.has_value() && conf.transpose_b()) {
      b_batch_axis.set_value(1 - b_batch_axis.value());
    }
    if (a_batch_axis.has_value() && a_batch_axis.value() == 0) {
      *BatchAxis4BnInOp("out") = a_batch_axis;
    } else if (b_batch_axis.has_value() && b_batch_axis.value() == 1) {
      *BatchAxis4BnInOp("out") = b_batch_axis;
    } else {
      BatchAxis4BnInOp("out")->clear_value();
    }
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  return Maybe<void>::Ok();
}

Maybe<void> MatmulOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const MatmulOpConf& conf = op_conf().matmul_conf();
  int32_t num_axes = JUST(LogicalBlobDesc4Ibn("a"))->shape().NumAxes();
  if (num_axes > 2) {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  } else if (num_axes == 2) {
    // (m, k_a) * (k_b, n) where k_a == k_b
    int32_t m_axis = -1;
    int32_t k_a_axis = -1;
    int32_t k_b_axis = -1;
    int32_t n_axis = -1;
    if (conf.transpose_a()) {
      m_axis = 1;
      k_a_axis = 0;
    } else {
      m_axis = 0;
      k_a_axis = 1;
    }
    if (conf.transpose_b()) {
      k_b_axis = 1;
      n_axis = 0;
    } else {
      k_b_axis = 0;
      n_axis = 1;
    }
    SbpSignatureBuilder()
        .Split("a", m_axis)
        .Broadcast("b")
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    SbpSignatureBuilder()
        .Broadcast("a")
        .Split("b", n_axis)
        .Split(output_bns(), 1)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    SbpSignatureBuilder()
        .Split("a", k_a_axis)
        .Split("b", k_b_axis)
        .PartialSum(output_bns())
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  } else {
    std::shared_ptr<ErrorProto> err;
    err->set_msg("MatMulOp: number of axis is " + std::to_string(num_axes) + " (not supported).");
    err->mutable_check_failed();
    return err;
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kMatmulConf, MatmulOp);

}  // namespace oneflow
