#include "oneflow/core/job_completer/job_completer.h"

namespace oneflow {

namespace {

void GenerateFacadeImplOpConf(const OpNode& op_node, const JobBuilder& job_builder) {
  CHECK(op_node.op().op_conf().has_print_scalar_summary_conf());
  const auto& print_scalar_summary_conf = op_node.op().op_conf().print_scalar_summary_conf();
  const LogicalBlobId& vector_lbi = op_node.op().BnInOp2Lbi("x");
  std::vector<OperatorConf> new_ops;
  // reduce sum
  OperatorConf reduce_sum_op_conf;
  reduce_sum_op_conf.set_name("System-Facade-" + op_node.op().op_name() + "_reduce_sum");
  auto* reduce_sum_conf = reduce_sum_op_conf.mutable_reduce_sum_conf();
  reduce_sum_conf->set_in(GenLogicalBlobName(vector_lbi));
  reduce_sum_conf->add_axis(0);
  reduce_sum_conf->set_out("out");
  new_ops.push_back(reduce_sum_op_conf);
  // get instance num by ShapeElemCntOp
  OperatorConf instance_num_op_conf;
  instance_num_op_conf.set_name("System-Facade-" + op_node.op().op_name() + "_instance_num");
  auto* instance_num_conf = instance_num_op_conf.mutable_shape_elem_cnt_conf();
  instance_num_conf->mutable_include_axis_conf()->add_axis(0);
  instance_num_conf->set_x(GenLogicalBlobName(vector_lbi));
  instance_num_conf->set_y("y");
  new_ops.push_back(instance_num_op_conf);
  LogicalBlobId scalar_value_lbi4print;
  LogicalBlobId instance_num_lbi4print;
  if (print_scalar_summary_conf.interval() > 1) {
    // accumulate scalar value
    OperatorConf scalar_value_acc_op_conf;
    scalar_value_acc_op_conf.set_name("System-Facade-" + op_node.op().op_name()
                                      + "_scalar_value_acc");
    auto* scalar_value_acc_sum_conf = scalar_value_acc_op_conf.mutable_acc_conf();
    scalar_value_acc_sum_conf->set_one(reduce_sum_op_conf.name() + "/out");
    scalar_value_acc_sum_conf->set_acc("acc");
    scalar_value_acc_sum_conf->set_max_acc_num(print_scalar_summary_conf.interval());
    new_ops.push_back(scalar_value_acc_op_conf);
    // accumulate instance_num
    OperatorConf instance_num_acc_op_conf;
    instance_num_acc_op_conf.set_name("System-Facade-" + op_node.op().op_name()
                                      + "_instance_num_acc");
    auto* instance_num_acc_conf = instance_num_acc_op_conf.mutable_acc_conf();
    instance_num_acc_conf->set_one(instance_num_op_conf.name() + "/y");
    instance_num_acc_conf->set_acc("acc");
    instance_num_acc_conf->set_max_acc_num(print_scalar_summary_conf.interval());
    new_ops.push_back(instance_num_acc_op_conf);
    scalar_value_lbi4print = GenLogicalBlobId(scalar_value_acc_op_conf.name() + "/acc");
    instance_num_lbi4print = GenLogicalBlobId(instance_num_acc_op_conf.name() + "/acc");
  } else {
    scalar_value_lbi4print = GenLogicalBlobId(reduce_sum_op_conf.name() + "/out");
    instance_num_lbi4print = GenLogicalBlobId(instance_num_op_conf.name() + "/y");
  }
  {
    OperatorConf instance_num_cast_op_conf;
    instance_num_cast_op_conf.set_name("System-Facade-" + op_node.op().op_name()
                                       + "_instance_num_cast");
    auto* instance_num_cast_conf = instance_num_cast_op_conf.mutable_cast_conf();
    instance_num_cast_conf->set_in(GenLogicalBlobName(instance_num_lbi4print));
    instance_num_cast_conf->set_out("out");
    instance_num_cast_conf->set_data_type(DataType::kFloat);
    new_ops.push_back(instance_num_cast_op_conf);
    instance_num_lbi4print = GenLogicalBlobId(instance_num_cast_op_conf.name() + "/out");
  }
  job_builder.AddOps(op_node.SoleInEdge()->src_node()->parallel_desc().parallel_conf(), new_ops);
  // scalar print
  OperatorConf print_op_conf;
  print_op_conf.set_name(op_node.op().op_name());
  auto* print_conf = print_op_conf.mutable_loss_print_conf();
  *(print_conf->mutable_loss_lbi()) = scalar_value_lbi4print;
  *(print_conf->mutable_loss_instance_num_lbi()) = instance_num_lbi4print;
  print_conf->set_reduction_type(print_scalar_summary_conf.reduction_type());
  print_conf->set_weight_scalar(print_scalar_summary_conf.weight_scalar());
  if (print_scalar_summary_conf.has_weight()) {
    *(print_conf->mutable_reduction_lbi()) = GenLogicalBlobId(print_scalar_summary_conf.weight());
  }
  job_builder.MutOps({print_op_conf});
}

}  // namespace

REGISTER_FACADE_IMPL(OperatorConf::kPrintScalarSummaryConf, &GenerateFacadeImplOpConf);

}  // namespace oneflow
