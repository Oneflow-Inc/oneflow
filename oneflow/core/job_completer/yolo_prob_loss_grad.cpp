#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_yolo_prob_loss_conf());
  if (DiffLbi4BnInOp("bbox_objness") != nullptr) {
    OperatorConf op_conf;
    op_conf.set_name(op.op_name() + "_bbox_objness_grad");
    MultiplyOpConf* conf = op_conf.mutable_multiply_conf();
    conf->set_out("out");

    conf->set_in_0(GenLogicalBlobName(*DiffLbi4BnInOp("bbox_objness_out")));
    conf->set_in_1(GenLogicalBlobName(op.BnInOp2Lbi("bbox_objness_out")));

    op_confs->push_back(op_conf);

    DiffLbi4BnInOp("bbox_objness")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("bbox_objness")->set_blob_name(conf->out());
  }
  if (DiffLbi4BnInOp("bbox_clsprob") != nullptr) {
    OperatorConf op_conf;
    op_conf.set_name(op.op_name() + "_bbox_clsprob_grad");
    MultiplyOpConf* conf = op_conf.mutable_multiply_conf();
    conf->set_out("out");

    conf->set_in_0(GenLogicalBlobName(*DiffLbi4BnInOp("bbox_clsprob_out")));
    conf->set_in_1(GenLogicalBlobName(op.BnInOp2Lbi("bbox_clsprob_out")));

    op_confs->push_back(op_conf);

    DiffLbi4BnInOp("bbox_clsprob")->set_op_name(op_conf.name());
    DiffLbi4BnInOp("bbox_clsprob")->set_blob_name(conf->out());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kYoloProbLossConf, &GenerateBackwardOpConf);

}  // namespace oneflow
