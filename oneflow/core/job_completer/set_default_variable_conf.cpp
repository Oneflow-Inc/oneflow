#include "oneflow/core/job_completer/set_default_variable_conf.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/job_set_compile_ctx.h"

namespace oneflow {

void SetDefaultVariableConf(const OpGraph& op_graph, JobBuilder* job_builder) {
  auto BlobDesc4ModelLbi = op_graph.MakeGetterBlobDesc4ModelLbi();
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (op_node->op().op_conf().has_variable_conf()) {
      OperatorConf variable_op_conf(op_node->op().op_conf());
      CHECK(job_builder->job().job_conf().has_default_data_type()
            || variable_op_conf.variable_conf().has_data_type());
      if (variable_op_conf.variable_conf().has_data_type() == false) {
        variable_op_conf.mutable_variable_conf()->set_data_type(
            job_builder->job().job_conf().default_data_type());
      }
      CHECK(job_builder->job().job_conf().has_default_initializer_conf()
            || variable_op_conf.variable_conf().has_initializer());
      if (variable_op_conf.variable_conf().has_initializer() == false) {
        *variable_op_conf.mutable_variable_conf()->mutable_initializer() =
            job_builder->job().job_conf().default_initializer_conf();
      }
      int64_t random_seed;
      auto* var_op_name2random = Global<JobSetCompileCtx>::Get()->GetVarOpName2randomSeed();
      const std::string& var_op_name = variable_op_conf.name();
      if (variable_op_conf.variable_conf().has_random_seed()) {
        random_seed = variable_op_conf.variable_conf().random_seed();
      } else {
        random_seed = NewRandomSeed();
      }
      const auto& pair = var_op_name2random->insert({var_op_name, random_seed});
      if (variable_op_conf.variable_conf().has_random_seed()) {
        CHECK_EQ(variable_op_conf.variable_conf().random_seed(), pair.first->second);
      } else {
        variable_op_conf.mutable_variable_conf()->set_random_seed(pair.first->second);
      }
      job_builder->AddOrMutOpsOnlyOnce(op_node->parallel_desc().parallel_conf(),
                                       {variable_op_conf});
    }
  });
}

}  // namespace oneflow
