#include "oneflow/core/autograd/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  // TODO: support optimizers other than SGD
  const auto& train_conf =
      Global<JobDesc>::Get()->other_conf().predict_conf().tmp_split_fw_bw_train_conf();
  const auto& model_update_conf = train_conf.model_update_conf();
  if (model_update_conf.has_naive_conf()) {
    // TODO complete regularization
    OperatorConf axpy_op;
    axpy_op.set_name(op.op_name() + "_grad");
    AxpyOpConf* axpy_op_conf = axpy_op.mutable_axpy_conf();
    axpy_op_conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    axpy_op_conf->set_y(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    axpy_op_conf->set_alpha(-train_conf.primary_lr());
    op_confs->push_back(axpy_op);
  } else if (model_update_conf.has_momentum_conf()) {
    TODO();
  } else if (model_update_conf.has_rmsprop_conf()) {
    TODO();
  } else if (model_update_conf.has_adam_conf()) {
    TODO();
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kVariableConf, &GenerateBackwardOpConf);

}  // namespace oneflow
