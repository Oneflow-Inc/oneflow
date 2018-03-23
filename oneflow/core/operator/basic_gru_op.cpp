#include "oneflow/core/operator/basic_gru_op.h"

namespace oneflow {

const PbMessage& BasicGruOp::GetCustomizedConf() const {
  return op_conf().basic_gru_conf();
}

void BasicGruOp::VirtualInitFromOpConf() {
#define OF_GRU_ENROLL_DATA_TMP_BN(modelname) EnrollDataTmpBn
#undef OF_GRU_ENROLL_DATA_TMP_BN
  EnrollDataTmpBn("reset_gate_data");
  EnrollModelBn("i2h_weight_r");
  EnrollModelBn("h2h_weight_r");
  EnrollDataTmpBn("reset_gate_out");
  EnrollDataTmpBn("reset_gate_data_diff");
  EnrollDataTmpBn("reset_gate_out_diff");

  EnrollDataTmpBn("update_gate_data");
  EnrollModelBn("i2h_weight_z");
  EnrollModelBn("h2h_weight_z");
  EnrollDataTmpBn("update_gate_out");
  EnrollDataTmpBn("update_gate_data_diff");
  EnrollDataTmpBn("update_gate_out_diff");

  EnrollDataTmpBn("candidate_hidden_data");
  EnrollModelBn("i2h_weight");
  EnrollModelBn("h2h_weight");
  EnrollDataTmpBn("candidate_hidden_out");
  EnrollDataTmpBn("candidate_hidden_data_diff");
  EnrollDataTmpBn("candidate_hidden_out_diff");

  EnrollDataTmpBn("temp_data");
  EnrollDataTmpBn("plus_op_out");

  if (GetBoolFromCustomizedConf("use_bias")) {
    EnrollModelBn("bias_r");
    EnrollModelTmpBn("bias_multiplier_r");

    EnrollModelBn("bias_z");
    EnrollModelTmpBn("bias_multiplier_z");

    EnrollModelBn("bias");
    EnrollModelTmpBn("bias_multiplier");
  }
}

void BasicGruOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  int32_t hidden_size = GetBlobDesc4BnInOp("out")->shape().At(1);
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int64_t embedding_size = in_blob_desc->shape().Count(1);
  int64_t data_num = in_blob_desc->shape().At(0);
#define OF_GRU_INFER_BLOB_DESCS(modelname)                                     \
  *GetBlobDesc4BnInOp(#modelname) = BlobDesc(                                  \
      Shape({data_num, hidden_size}), JobDesc::Singleton()->DefaultDataType(), \
      false, true, in_blob_desc->max_col_num())
  OF_GRU_INFER_BLOB_DESCS(reset_gate_data);
  OF_GRU_INFER_BLOB_DESCS(reset_gate_out);
  OF_GRU_INFER_BLOB_DESCS(update_gate_data);
  OF_GRU_INFER_BLOB_DESCS(update_gate_out);
  OF_GRU_INFER_BLOB_DESCS(candidate_hidden_data);
  OF_GRU_INFER_BLOB_DESCS(candidate_hidden_out);
  OF_GRU_INFER_BLOB_DESCS(temp_data);
  OF_GRU_INFER_BLOB_DESCS(plus_op_out);
#undef OF_GRU_INFER_BLOB_DESCS

#define OF_GRU_INFER_WEIGHT_DESCS(i2h_weight, h2h_weight) \
  *GetBlobDesc4BnInOp(#i2h_weight) =                      \
      BlobDesc(Shape({hidden_size, embedding_size}));     \
  *GetBlobDesc4BnInOp(#h2h_weight) = BlobDesc(Shape({hidden_size, hidden_size}))
  OF_GRU_INFER_WEIGHT_DESCS(i2h_weight_r, h2h_weight_r);
  OF_GRU_INFER_WEIGHT_DESCS(i2h_weight_z, h2h_weight_z);
  OF_GRU_INFER_WEIGHT_DESCS(i2h_weight, h2h_weight);
#undef OF_GRU_INFER_WEIGHT_DESCS

  if (GetBoolFromCustomizedConf("use_bias")) {
    *GetBlobDesc4BnInOp("bias_r") = BlobDesc(Shape({1, hidden_size}));
    *GetBlobDesc4BnInOp("bias_multiplier_r") = BlobDesc(Shape({data_num, 1}));

    *GetBlobDesc4BnInOp("bias_z") = BlobDesc(Shape({1, hidden_size}));
    *GetBlobDesc4BnInOp("bias_multiplier_z") = BlobDesc(Shape({data_num, 1}));

    *GetBlobDesc4BnInOp("bias") = BlobDesc(Shape({1, hidden_size}));
    *GetBlobDesc4BnInOp("bias_multiplier") = BlobDesc(Shape({data_num, 1}));
  }
}

REGISTER_OP(OperatorConf::kBasicGruConf, BasicGruOp);

}  // namespace oneflow
