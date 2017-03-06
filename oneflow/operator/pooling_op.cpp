#include "operator/pooling_op.h"
#include "glog/logging.h"

namespace oneflow {

void PoolingOp::Init(const OperatorConf& op_conf) {
  mutable_op_name() = op_conf.name();
  
  CHECK(op_conf.has_pooling_op_conf());
  auto cnf_ptr = new PoolingOpConf(op_conf.pooling_op_conf());
  mutable_pb_op_conf().reset(cnf_ptr);

  auto data_ptr = new PoolingDataBlobDescSet();
  data_ptr->Init(op_name());
  mutable_data_blob_desc_set().reset(data_ptr);

  auto model_ptr = new PoolingModelBlobDescSet();
  model_ptr->Init(op_name());
  mutable_model_blob_desc_set().reset(model_ptr);
}

} // namespace oneflow
