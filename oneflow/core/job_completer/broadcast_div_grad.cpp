#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_broadcast_div_conf());
  if (DiffLbi4BnInOp("a") != nullptr) {
    OperatorConf broadcast_div_a;
    broadcast_div_a.set_name(op.op_name() + "_grad_a_div");
    BroadcastDivOpConf* broadcast_div_a_op_conf = broadcast_div_a.mutable_broadcast_div_conf();
    broadcast_div_a_op_conf->set_a(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    broadcast_div_a_op_conf->set_b(GenLogicalBlobName(op.BnInOp2Lbi("b")));
    broadcast_div_a_op_conf->set_out("out");
    op_confs->push_back(broadcast_div_a);

    if (LogicalBlobDesc4BnInOp("out").shape() == LogicalBlobDesc4BnInOp("a").shape()) {
      DiffLbi4BnInOp("a")->set_op_name(broadcast_div_a.name());
      DiffLbi4BnInOp("a")->set_blob_name("out");
    } else {
      OperatorConf reduce_sum_like_a;
      reduce_sum_like_a.set_name(op.op_name() + "_grad_a_reduce");
      ReduceSumLikeOpConf* reduce_sum_like_a_op_conf =
          reduce_sum_like_a.mutable_reduce_sum_like_conf();
      reduce_sum_like_a_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("a")));
      reduce_sum_like_a_op_conf->set_x(broadcast_div_a.name() + "/out");
      reduce_sum_like_a_op_conf->set_y("y");
      const std::vector<int64_t> broadcast_axis_vec =
          Shape::AxisByBroadcastTo(LogicalBlobDesc4BnInOp("a").shape().CreateLeftExtendedShape(
                                       LogicalBlobDesc4BnInOp("out").shape()),
                                   LogicalBlobDesc4BnInOp("out").shape());
      *reduce_sum_like_a_op_conf->mutable_axis() = {broadcast_axis_vec.begin(),
                                                    broadcast_axis_vec.end()};

      op_confs->push_back(reduce_sum_like_a);
      DiffLbi4BnInOp("a")->set_op_name(reduce_sum_like_a.name());
      DiffLbi4BnInOp("a")->set_blob_name("y");
    }
  }
  if (DiffLbi4BnInOp("b") != nullptr) {
    OperatorConf broadcast_div_grad_op;
    broadcast_div_grad_op.set_name(op.op_name() + "_grad");
    BroadcastDivGradOpConf* broadcast_div_grad_op_conf =
        broadcast_div_grad_op.mutable_broadcast_div_grad_conf();
    broadcast_div_grad_op_conf->set_b(GenLogicalBlobName(op.BnInOp2Lbi("b")));
    broadcast_div_grad_op_conf->set_y(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    broadcast_div_grad_op_conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    broadcast_div_grad_op_conf->set_db("db");
    op_confs->push_back(broadcast_div_grad_op);
    DiffLbi4BnInOp("b")->set_op_name(broadcast_div_grad_op.name());
    DiffLbi4BnInOp("b")->set_blob_name("db");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kBroadcastDivConf, &GenerateBackwardOpConf);

}  // namespace oneflow
