#include "layer/multinomial_logistic_loss_layer_desc.h"
#include "glog/logging.h"

namespace oneflow {


void MLLossDataBlobDescSet::Init(const std::string& layer_name) {
  DataBlobDescSet::Init();
  RegisterInputBlobPptr(layer_name + "/data", &data_);
  RegisterInputDiffBlobPptr(layer_name + "/data_diff", &data_diff_);
  RegisterInputBlobPptr(layer_name + "/label", &label_);
  RegisterInputDiffBlobPptr(layer_name + "/label_diff", &label_diff_);
  RegisterOutputBlobPptr(layer_name + "/loss", &loss_);
  RegisterDataTmpBlobPptr(layer_name + "/loss_buffer", &loss_buffer_);
}

void MultinomialLogisticLossLayerDesc::Init(const LayerConf& layer_conf) {
  mutable_layer_name() = layer_conf.name();
  
  CHECK(layer_conf.has_multinomial_logistic_loss_layer_conf());
  auto cnf_ptr =
      new MultinomialLogisticLossLayerConf(layer_conf.multinomial_logistic_loss_layer_conf());
  mutable_pb_layer_conf().reset(cnf_ptr);

  auto data_ptr = new MLLossDataBlobDescSet();
  data_ptr->Init(layer_name());
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new MLLossModelBlobDescSet();
  model_ptr->Init(layer_name());
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
