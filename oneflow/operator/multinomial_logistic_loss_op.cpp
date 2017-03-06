#include "operator/multinomial_logistic_loss_op.h"
#include "glog/logging.h"

namespace oneflow {

void MLLossDataBlobDescSet::Init(const std::string& op_name) {
  DataBlobDescSet::Init();
  RegisterInputBlobPptr(op_name + "/data", &data_);
  RegisterInputDiffBlobPptr(op_name + "/data_diff", &data_diff_);
  RegisterInputBlobPptr(op_name + "/label", &label_);
  RegisterInputDiffBlobPptr(op_name + "/label_diff", &label_diff_);
  RegisterOutputBlobPptr(op_name + "/loss", &loss_);
  RegisterDataTmpBlobPptr(op_name + "/loss_buffer", &loss_buffer_);
}

void MultinomialLogisticLossOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_multinomial_logistic_loss_op_conf());
  auto cnf_ptr =
      new MultinomialLogisticLossOpConf(
          op_conf.multinomial_logistic_loss_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);

  auto data_ptr = new MLLossDataBlobDescSet();
  data_ptr->Init(op_name());
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new MLLossModelBlobDescSet();
  model_ptr->Init(op_name());
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
