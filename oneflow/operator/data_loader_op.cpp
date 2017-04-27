#include "operator/data_loader_op.h"
#include "glog/logging.h"
#include "operator/operator_factory.h"

namespace oneflow {

void DataLoaderOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_data_loader_conf());
  mut_op_conf() = op_conf;
 
  EnrollOutputBn("data", false);
  EnrollOutputBn("label", false);
}

std::string DataLoaderOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().data_loader_conf(), k);
}

REGISTER_OP(OperatorConf::kDataLoaderConf, DataLoaderOp);

} // namespace oneflow
