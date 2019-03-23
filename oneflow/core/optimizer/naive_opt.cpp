#include "oneflow/core/optimizer/optimizer.h"

namespace oneflow {

namespace {

void GenerateOptimizerOpConf(
    const VariableOp& op, std::vector<OperatorConf>* op_confs,
    const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp,
    const LogicalBlobId& total_loss_instance_num_lbi) {
  op_confs->push_back(ConstructMdUpdtOpConf(op, DiffLbi4BnInOp, total_loss_instance_num_lbi));
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kNaiveConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
