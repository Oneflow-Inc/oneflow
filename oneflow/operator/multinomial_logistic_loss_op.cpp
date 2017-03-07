#include "operator/multinomial_logistic_loss_op.h"
#include "glog/logging.h"

namespace oneflow {

void MLLossDataBlobDescSet::Init() {
  DataBlobDescSet::Init();
  RegisterInputBlobPptr("data", &data_);
  RegisterInputDiffBlobPptr("data_diff", &data_diff_);
  RegisterInputBlobPptr("label", &label_);
  RegisterInputDiffBlobPptr("label_diff", &label_diff_);
  RegisterOutputBlobPptr("loss", &loss_);
  RegisterDataTmpBlobPptr("loss_buffer", &loss_buffer_);
}

void MultinomialLogisticLossOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_multinomial_logistic_loss_op_conf());
  auto cnf_ptr =
      new MultinomialLogisticLossOpConf(
          op_conf.multinomial_logistic_loss_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);

  auto data_ptr = new MLLossDataBlobDescSet();
  data_ptr->Init();
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new MLLossModelBlobDescSet();
  model_ptr->Init();
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
