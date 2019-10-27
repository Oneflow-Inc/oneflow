#include "oneflow/core/job_completer/job_completer.h"
#include "oneflow/core/register/dense_shape_view.h"

namespace oneflow {

namespace {

void UpdateConsumerOpConf(const std::string& new_lbn, const OpNode& op_node,
                          JobBuilder* job_builder) {
  const LogicalBlobId& old_lbi = op_node.op().BnInOp2Lbi(op_node.op().SoleObn());
  const std::string& old_lbn = GenLogicalBlobName(old_lbi);
  for (const OpEdge* edge : op_node.out_edges()) {
    OpNode* out_node = edge->dst_node();
    OperatorConf mut_op_conf(out_node->op().op_conf());
    PbMessage* mut_conf = MutableMessageInPbMessage(&mut_op_conf, mut_op_conf.op_type_case());
    for (const std::string& ibn : edge->lbi2ibns().at(old_lbi)) {
      ReplaceStrValInPbFdOrPbRpf(mut_conf, ibn, old_lbn, new_lbn);
    }
    job_builder->MutOpsOnlyOnce({mut_op_conf});
  }
}

void GenerateFacadeImplOpConf(const OpNode& op_node, JobBuilder* job_builder) {
  CHECK(op_node.op().op_conf().has_reduce_mean_conf());
  const auto& reduce_mean_conf = op_node.op().op_conf().reduce_mean_conf();
  OperatorConf reduce_sum_op_conf(op_node.op().op_conf());
  auto* reduce_sum_conf = reduce_sum_op_conf.mutable_reduce_sum_conf();
  reduce_sum_conf->set_in(reduce_mean_conf.in());
  *reduce_sum_conf->mutable_axis() = reduce_mean_conf.axis();
  reduce_sum_conf->set_out("out");
  job_builder->MutOpsOnlyOnce({reduce_sum_op_conf});

  const auto& in_blob = op_node.LogicalBlobDesc4Lbi(GenLogicalBlobId(reduce_mean_conf.in()));
  CHECK_EQ(in_blob.num_of_lod_levels(), 0);
  std::string out_lbi;
  if (in_blob.is_dynamic()) {  // TODO: && in_blob.has_instance_shape()
    OperatorConf partial_elem_cnt_op_conf;
    partial_elem_cnt_op_conf.set_name("System-Facade-" + op_node.op().op_name()
                                      + "_partial_elem_cnt");
    auto* partial_elem_cnt_conf = partial_elem_cnt_op_conf.mutable_shape_elem_cnt_conf();
    if (reduce_mean_conf.axis().empty()) {
      partial_elem_cnt_conf->mutable_exclude_axis_conf();
    } else {
      *partial_elem_cnt_conf->mutable_include_axis_conf()->mutable_axis() = reduce_mean_conf.axis();
    }
    partial_elem_cnt_conf->set_x(reduce_mean_conf.in());
    partial_elem_cnt_conf->set_y("y");
    partial_elem_cnt_conf->set_data_type(
        op_node.LogicalBlobDesc4Lbi(op_node.op().BnInOp2Lbi(op_node.op().SoleObn())).data_type());

    OperatorConf boradcast_div_op_conf;
    boradcast_div_op_conf.set_name("System-Facade-" + op_node.op().op_name() + "_broadcast_div");
    auto* boradcast_div_conf = boradcast_div_op_conf.mutable_broadcast_div_conf();
    boradcast_div_conf->set_a(reduce_sum_op_conf.name() + "/out");
    boradcast_div_conf->set_b(partial_elem_cnt_op_conf.name() + "/y");
    boradcast_div_conf->set_out("out");
    job_builder->AddOps(op_node.parallel_desc().parallel_conf(),
                        {partial_elem_cnt_op_conf, boradcast_div_op_conf});
    out_lbi = boradcast_div_op_conf.name() + "/out";
  } else {
    double shape_elem_cnt = 1.0;
    if (reduce_mean_conf.axis().empty()) {
      shape_elem_cnt = in_blob.shape().elem_cnt();
    } else {
      const auto& axes = reduce_mean_conf.axis();
      for (const auto& axis : axes) {
        shape_elem_cnt *= in_blob.shape().At(ShiftNegativeAxis(axis, in_blob.shape().NumAxes()));
      }
    }
    OperatorConf scalar_div_op_conf;
    scalar_div_op_conf.set_name("System-Facade-" + op_node.op().op_name() + "_scalar_div");
    auto* scalar_div_conf = scalar_div_op_conf.mutable_scalar_mul_conf();
    scalar_div_conf->set_in(reduce_sum_op_conf.name() + "/out");
    scalar_div_conf->set_out("out");
    scalar_div_conf->set_float_operand(1 / shape_elem_cnt);
    job_builder->AddOps(op_node.parallel_desc().parallel_conf(), {scalar_div_op_conf});
    out_lbi = scalar_div_op_conf.name() + "/out";
  }
  UpdateConsumerOpConf(out_lbi, op_node, job_builder);
}

}  // namespace

REGISTER_FACADE_IMPL(OperatorConf::kReduceMeanConf, &GenerateFacadeImplOpConf);

}  // namespace oneflow
