#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_local_scatter_nd_update_conf());
  // updates diff
  if (DiffLbi4BnInOp("updates") != nullptr) {
    OperatorConf updates_grad_op_conf;
    updates_grad_op_conf.set_name(op.op_name() + "-updates_diff");
    auto* conf = updates_grad_op_conf.mutable_local_gather_nd_conf();
    conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_indices(GenLogicalBlobName(op.BnInOp2Lbi("indices")));
    conf->set_out("out");
    op_confs->push_back(updates_grad_op_conf);
    DiffLbi4BnInOp("updates")->set_op_name(updates_grad_op_conf.name());
    DiffLbi4BnInOp("updates")->set_blob_name(conf->out());
  }
  // in diff
  if (DiffLbi4BnInOp("in") != nullptr) {
    // constant zero updates
    OperatorConf zero_updates_op_conf;
    zero_updates_op_conf.set_name(op.op_name() + "-backward_zero_updates");
    auto* zu_conf = zero_updates_op_conf.mutable_constant_conf();
    LogicalBlobDesc4BnInOp("updates").shape().ToProto(zu_conf->mutable_shape());
    zu_conf->set_data_type(LogicalBlobDesc4BnInOp("updates").data_type());
    zu_conf->mutable_initializer()->mutable_constant_conf()->set_value(0.0f);
    zu_conf->set_out("out");
    op_confs->push_back(zero_updates_op_conf);
    LogicalBlobId zu_lbi;
    zu_lbi.set_op_name(zero_updates_op_conf.name());
    zu_lbi.set_blob_name(zu_conf->out());

    OperatorConf in_grad_op_conf;
    in_grad_op_conf.set_name(op.op_name() + "-in_diff");
    auto* conf = in_grad_op_conf.mutable_local_scatter_nd_update_conf();
    conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_indices(GenLogicalBlobName(op.BnInOp2Lbi("indices")));
    conf->set_updates(GenLogicalBlobName(zu_lbi));
    conf->set_out("out");
    op_confs->push_back(in_grad_op_conf);
    DiffLbi4BnInOp("in")->set_op_name(in_grad_op_conf.name());
    DiffLbi4BnInOp("in")->set_blob_name(conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kLocalScatterNdUpdateConf, &GenerateBackwardOpConf);

}  // namespace oneflow
