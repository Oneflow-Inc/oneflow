#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  const auto& train_conf =
      Global<JobDesc>::Get()->other_conf().predict_conf().tmp_split_fw_bw_train_conf();
  const auto& model_update_conf = train_conf.model_update_conf();
  OperatorConf mdupdt_op;
  mdupdt_op.set_name(op.op_name() + "_grad");
  mdupdt_op.mutable_normal_mdupdt_conf()->set_model_diff(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
  mdupdt_op.mutable_normal_mdupdt_conf()->set_total_instance_num_diff(op.op_name() + '/'
                                                                      + "total_instance_num");
  mdupdt_op.mutable_normal_mdupdt_conf()->set_model(GenLogicalBlobName(op.BnInOp2Lbi("out")));
  if (op.op_conf().variable_conf().ModelName() == "total_instance_num") {
    mdupdt_op.mutable_normal_mdupdt_conf()->mutable_user_conf()->mutable_naive_conf();
  } else {
    *(mdupdt_op.mutable_normal_mdupdt_conf()->mutable_user_conf()) =
        Global<JobDesc>::Get()->other_conf().train_conf().model_update_conf();
  }
  float primary_lr = Global<JobDesc>::Get()->primary_lr();
  float secondary_lr = Global<JobDesc>::Get()->secondary_lr();
  if (secondary_lr < 0) { secondary_lr = primary_lr; }
  if (op.op_conf().variable_conf().ModelName() == "weight") {
    mdupdt_op.mutable_normal_mdupdt_conf()->set_learning_rate(primary_lr);
    mdupdt_op.mutable_normal_mdupdt_conf()->set_l1(Global<JobDesc>::Get()->weight_l1());
    mdupdt_op.mutable_normal_mdupdt_conf()->set_l2(Global<JobDesc>::Get()->weight_l2());
  } else if (op.op_conf().variable_conf().ModelName() == "bias") {
    mdupdt_op.mutable_normal_mdupdt_conf()->set_learning_rate(secondary_lr);
    mdupdt_op.mutable_normal_mdupdt_conf()->set_l1(Global<JobDesc>::Get()->bias_l1());
    mdupdt_op.mutable_normal_mdupdt_conf()->set_l2(Global<JobDesc>::Get()->bias_l2());
  } else if (op.op_conf().variable_conf().ModelName() == "total_instance_num") {
    // we don't treat total_instance_num as model, just use total_instance_num_diff
    mdupdt_op.mutable_normal_mdupdt_conf()->set_learning_rate(-1.0);
    mdupdt_op.mutable_normal_mdupdt_conf()->set_l1(0);
    mdupdt_op.mutable_normal_mdupdt_conf()->set_l2(0);
  } else {
    mdupdt_op.mutable_normal_mdupdt_conf()->set_learning_rate(primary_lr);
    mdupdt_op.mutable_normal_mdupdt_conf()->set_l1(0);
    mdupdt_op.mutable_normal_mdupdt_conf()->set_l2(0);
  }
  op_confs->push_back(mdupdt_op);
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kVariableConf, &GenerateBackwardOpConf);

}  // namespace oneflow
