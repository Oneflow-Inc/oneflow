#include "oneflow/core/operator/basic_gru_op.h"

namespace oneflow {

const PbMessage& BasicGruOp::GetSpecialConf() const {
  return op_conf().basic_gru_conf();
}

void BasicGruOp::VirtualInitFromOpConf() {
  EnrollDataTmpBn("state_data");

  EnrollDataTmpBn("reset_out");
  EnrollModelBn("i2h_r_weight");
  EnrollModelBn("h2h_r_weight");
  EnrollDataTmpBn("reset_data_diff");
  EnrollDataTmpBn("reset_out_diff");

  EnrollDataTmpBn("update_out");
  EnrollModelBn("i2h_z_weight");
  EnrollModelBn("h2h_z_weight");
  EnrollDataTmpBn("update_data_diff");
  EnrollDataTmpBn("update_out_diff");

  EnrollDataTmpBn("candidate_out");
  EnrollModelBn("i2h_weight");
  EnrollModelBn("h2h_weight");
  EnrollDataTmpBn("candidate_data_diff");
  EnrollDataTmpBn("candidate_out_diff");

  EnrollDataTmpBn("tmp_data");

  EnrollModelBn("bias_r");
  EnrollModelTmpBn("bias_r_multiplier");

  EnrollModelBn("bias_z");
  EnrollModelTmpBn("bias_z_multiplier");

  EnrollModelBn("bias");
  EnrollModelTmpBn("bias_multiplier");
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
  OF_GRU_INFER_BLOB_DESCS(state_data);
  OF_GRU_INFER_BLOB_DESCS(reset_out);
  OF_GRU_INFER_BLOB_DESCS(reset_data_diff);
  OF_GRU_INFER_BLOB_DESCS(reset_out_diff);
  OF_GRU_INFER_BLOB_DESCS(update_out);
  OF_GRU_INFER_BLOB_DESCS(update_data_diff);
  OF_GRU_INFER_BLOB_DESCS(update_out_diff);
  OF_GRU_INFER_BLOB_DESCS(candidate_out);
  OF_GRU_INFER_BLOB_DESCS(candidate_data_diff);
  OF_GRU_INFER_BLOB_DESCS(candidate_out_diff);
#undef OF_GRU_INFER_BLOB_DESCS

#define OF_GRU_INFER_WEIGHT_DESCS(i2h_weight, h2h_weight) \
  *GetBlobDesc4BnInOp(#i2h_weight) =                      \
      BlobDesc(Shape({hidden_size, embedding_size}));     \
  *GetBlobDesc4BnInOp(#h2h_weight) = BlobDesc(Shape({hidden_size, hidden_size}))
  OF_GRU_INFER_WEIGHT_DESCS(i2h_r_weight, h2h_r_weight);
  OF_GRU_INFER_WEIGHT_DESCS(i2h_z_weight, h2h_z_weight);
  OF_GRU_INFER_WEIGHT_DESCS(i2h_weight, h2h_weight);
#undef OF_GRU_INFER_WEIGHT_DESCS

  *GetBlobDesc4BnInOp("bias_r") = BlobDesc(Shape({1, hidden_size}));
  *GetBlobDesc4BnInOp("bias_r_multiplier") = BlobDesc(Shape({data_num, 1}));

  *GetBlobDesc4BnInOp("bias_z") = BlobDesc(Shape({1, hidden_size}));
  *GetBlobDesc4BnInOp("bias_z_multiplier") = BlobDesc(Shape({data_num, 1}));

  *GetBlobDesc4BnInOp("bias") = BlobDesc(Shape({1, hidden_size}));
  *GetBlobDesc4BnInOp("bias_multiplier") = BlobDesc(Shape({data_num, 1}));
}

REGISTER_OP(OperatorConf::kBasicGruConf, BasicGruOp);

}  // namespace oneflow
