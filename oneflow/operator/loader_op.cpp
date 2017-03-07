#include "operator/loader_op.h"
#include "glog/logging.h"

namespace oneflow {

void LoaderOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_loader_op_conf());
  auto cnf_ptr = new LoaderOpConf(op_conf.loader_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);
  
  auto data_ptr = new LoaderDataBlobDescSet();
  data_ptr->Init();
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new LoaderModelBlobDescSet();
  model_ptr->Init();
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
