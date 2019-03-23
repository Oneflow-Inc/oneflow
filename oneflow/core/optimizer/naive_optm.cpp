#include "oneflow/core/optimizer/optimizer.h"

namespace oneflow {

namespace {

void GenerateOptimizerOpConf(
    const VariableOp& op, std::vector<OperatorConf>* op_confs,
    const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp,
    const LogicalBlobId& total_loss_instance_num_lbi) {
  OperatorConf mdupdt_op;
  mdupdt_op.set_name(op.op_name() + "_optimizer");
  ConstructMdUpdtOpConf(op, DiffLbi4BnInOp, total_loss_instance_num_lbi,
                        mdupdt_op.mutable_naive_model_update_conf());
  op_confs->push_back(mdupdt_op);
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kNaiveConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
