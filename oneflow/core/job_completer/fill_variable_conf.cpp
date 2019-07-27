#include "oneflow/core/job_completer/fill_variable_conf.h"
#include "oneflow/core/job/job_builder.h"

namespace oneflow {

void FillVariableConf(const OpGraph& op_graph, Job* job) {
  auto BlobDesc4ModelLbi = op_graph.MakeGetterBlobDesc4ModelLbi();
  JobBuilder job_builder(job);
  InitializerConf default_initializer_conf;
  if (job->job_conf().has_default_initializer_conf()) {
    default_initializer_conf = job->job_conf().default_initializer_conf();
  } else {
    default_initializer_conf.mutable_xavier_conf();
    default_initializer_conf.mutable_xavier_conf()->set_variance_norm(kAverage);
  }
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (op_node->op().op_conf().has_variable_conf()) {
      OperatorConf variable_op_conf(op_node->op().op_conf());
      variable_op_conf.mutable_variable_conf()->set_data_type(job->job_conf().default_data_type());
      if (variable_op_conf.variable_conf().has_initializer() == false) {
        *variable_op_conf.mutable_variable_conf()->mutable_initializer() = default_initializer_conf;
      }
      job_builder.AddOrMutOpsOnlyOnce(op_node->parallel_desc().parallel_conf(), {variable_op_conf});
    }
  });
}

}  // namespace oneflow
