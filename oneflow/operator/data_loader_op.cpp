#include "operator/data_loader_op.h"
#include "glog/logging.h"
#include "operator/operator_manager.h"

namespace oneflow {

void DataLoaderOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_data_loader_conf());
  mut_op_conf() = op_conf;
 
  EnrollOutputBn("data", false);
  EnrollOutputBn("label", false);
}

const PbMessage& DataLoaderOp::GetSpecialConf() const {
  return op_conf().data_loader_conf();
}

REGISTER_OP(OperatorConf::kDataLoaderConf, DataLoaderOp);

} // namespace oneflow
