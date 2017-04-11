#include "operator/data_loader_op.h"
#include "glog/logging.h"

namespace oneflow {

void LoaderOp::Init(const OperatorConf& op_conf) {
  mut_op_name() = op_conf.name();
  
  CHECK(op_conf.has_loader_op_conf());
  auto cnf = new LoaderOpConf(op_conf.loader_op_conf());
  mut_pb_op_conf().reset(cnf);
 
  EnrollOutputBn("data");
  EnrollOutputBn("label");
}

} // namespace oneflow
