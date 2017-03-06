#include "operator/innerproduct_op.h"
#include "glog/logging.h"

namespace oneflow {

void InnerProductDataBlobDescSet::Init(const std::string& op_name) {
  RegisterInputBlobPptr(op_name + "/in", &in_);
  RegisterInputDiffBlobPptr(op_name + "/in_diff", &in_diff_);
  RegisterOutputBlobPptr(op_name + "/out", &out_);
  RegisterOutputDiffBlobPptr(op_name + "/out_diff", &out_diff_);
}

void InnerProductModelBlobDescSet::Init(const std::string& op_name) {
  RegisterModelBlobPptr(op_name + "/weight", &weight_);
  RegisterModelDiffBlobPptr(op_name + "/weight_diff", &weight_diff_);
  RegisterModelBlobPptr(op_name + "/bias", &bias_);
  RegisterModelDiffBlobPptr(op_name + "/bias_diff_", &bias_diff_);
  RegisterModelTmpBlobPptr(op_name + "/bias_multiplier",
                           &bias_multiplier_);
}

void InnerProductOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_inner_product_op_conf());
  auto cnf_ptr =
      new InnerProductOpConf(op_conf.inner_product_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);

  auto data_ptr = new InnerProductDataBlobDescSet();
  data_ptr->Init(op_name());
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new InnerProductModelBlobDescSet();
  model_ptr->Init(op_name());
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
