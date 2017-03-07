#include "operator/innerproduct_op.h"
#include "glog/logging.h"

namespace oneflow {

void InnerProductDataBlobDescSet::Init() {
  RegisterInputBlobPptr("in", &in_);
  RegisterInputDiffBlobPptr("in_diff", &in_diff_);
  RegisterOutputBlobPptr("out", &out_);
  RegisterOutputDiffBlobPptr("out_diff", &out_diff_);
}

void InnerProductModelBlobDescSet::Init() {
  RegisterModelBlobPptr("weight", &weight_);
  RegisterModelDiffBlobPptr("weight_diff", &weight_diff_);
  RegisterModelBlobPptr("bias", &bias_);
  RegisterModelDiffBlobPptr("bias_diff_", &bias_diff_);
  RegisterModelTmpBlobPptr("bias_multiplier", &bias_multiplier_);
}

void InnerProductOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_inner_product_op_conf());
  auto cnf_ptr =
      new InnerProductOpConf(op_conf.inner_product_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);

  auto data_ptr = new InnerProductDataBlobDescSet();
  data_ptr->Init();
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new InnerProductModelBlobDescSet();
  model_ptr->Init();
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
