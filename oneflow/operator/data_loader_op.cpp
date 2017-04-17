#include "operator/data_loader_op.h"
#include "glog/logging.h"

namespace oneflow {

void DataLoaderOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();
  
  CHECK(op_conf.has_data_loader_op_conf());
  auto cnf = new DataLoaderOpConf(op_conf.data_loader_op_conf());
  mut_pb_op_conf().reset(cnf);
 
  EnrollOutputBn("data", false);
  EnrollOutputBn("label", false);
}

} // namespace oneflow
