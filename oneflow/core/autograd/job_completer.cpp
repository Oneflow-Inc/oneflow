#include "oneflow/core/autograd/job_completer.h"
#include "oneflow/core/autograd/autovar.h"
#include "oneflow/core/autograd/autograd.h"
#include "oneflow/core/autograd/autotick.h"
#include "oneflow/core/optimizer/optimizer.h"

namespace oneflow {

namespace {

void WithOpGraphAndMutJobConf(const std::function<void(const OpGraph&, JobConf1*)>& Handler) {
  const JobDesc* job_desc = Global<JobDesc>::Get();
  OpGraph op_graph(job_desc);
  JobConf1 job_conf(job_desc->job_conf());
  Handler(op_graph, &job_conf);
  Global<JobDesc>::Delete();
  Global<JobDesc>::New(job_conf);
}

void GenerateFacadeImplOpConfIf(const OpNode& op_node, const JobConfBuilder& job_conf_builder) {
  auto op_type_case = op_node.op().op_conf().op_type_case();
  if (IsClassRegistered<GenerateFacadeImplOpConfWrapperStruct>(op_type_case)) {
    auto* obj = NewObj<GenerateFacadeImplOpConfWrapperStruct>(op_type_case);
    obj->Call(op_node, job_conf_builder);
  }
}

void ReplaceFacade(const OpGraph& op_graph, JobConf1* job_conf) {
  JobConfBuilder job_conf_builder(job_conf);
  op_graph.ForEachNode(
      [&](OpNode* op_node) { GenerateFacadeImplOpConfIf(*op_node, job_conf_builder); });
}

}  // namespace

void JobCompleter::CompleteGlobalJobDesc() const {
  // replace facade op
  WithOpGraphAndMutJobConf(&ReplaceFacade);
  // complete variable ops
  WithOpGraphAndMutJobConf(&AutoVar);
  // complete ops for trainning
  WithOpGraphAndMutJobConf([](const OpGraph& op_graph, JobConf1* job_conf) {
    LogicalBlobId total_loss_instance_num;
    AddTotalLossInstanceNumOpConf(op_graph, job_conf, &total_loss_instance_num);
    HashMap<LogicalBlobId, LogicalBlobId> lbi2diff_lbi;
    AutoGrad(op_graph, job_conf, &lbi2diff_lbi);
    AddOptimizerOpConf(op_graph, job_conf, lbi2diff_lbi, total_loss_instance_num);
  });
  // complete tick ops
  WithOpGraphAndMutJobConf(&AutoTick);
}

}  // namespace oneflow
