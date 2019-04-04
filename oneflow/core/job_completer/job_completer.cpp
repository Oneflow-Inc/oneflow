#include "oneflow/core/job_completer/job_completer.h"
#include "oneflow/core/job_completer/autovar.h"
#include "oneflow/core/job_completer/autograd.h"
#include "oneflow/core/job_completer/autotick.h"
#include "oneflow/core/job_completer/auto_saver.h"
#include "oneflow/core/optimizer/optimizer.h"

namespace oneflow {

namespace {

void WithOpGraphAndMutJob(const std::function<void(const OpGraph&, Job*)>& Handler) {
  const JobDesc* job_desc = Global<JobDesc>::Get();
  OpGraph op_graph(job_desc);
  Job job(job_desc->job());
  Handler(op_graph, &job);
  Global<JobDesc>::Delete();
  Global<JobDesc>::New(job);
}

void GenerateFacadeImplOpConfIf(const OpNode& op_node, const JobBuilder& job_builder) {
  auto op_type_case = op_node.op().op_conf().op_type_case();
  if (IsClassRegistered<GenerateFacadeImplOpConfWrapperStruct>(op_type_case)) {
    auto* obj = NewObj<GenerateFacadeImplOpConfWrapperStruct>(op_type_case);
    obj->Call(op_node, job_builder);
  }
}

void ReplaceFacade(const OpGraph& op_graph, Job* job) {
  JobBuilder job_builder(job);
  op_graph.ForEachNode([&](OpNode* op_node) { GenerateFacadeImplOpConfIf(*op_node, job_builder); });
}

}  // namespace

void JobCompleter::CompleteGlobalJobDesc() const {
  // replace facade op
  WithOpGraphAndMutJob(&ReplaceFacade);
  if (Global<JobDesc>::Get()->IsPredict()
      && Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    // complete variable ops
    WithOpGraphAndMutJob(&AutoVar);
    // complete ops for trainning
    WithOpGraphAndMutJob([](const OpGraph& op_graph, Job* job) {
      LogicalBlobId total_loss_instance_num;
      AddTotalLossInstanceNumOpConf(op_graph, job, &total_loss_instance_num);
      HashMap<LogicalBlobId, LogicalBlobId> lbi2diff_lbi;
      AutoGrad(op_graph, job, &lbi2diff_lbi);
      AddOptimizerOpConf(op_graph, job, lbi2diff_lbi, total_loss_instance_num);
      AutoSaver(op_graph, job);
    });
    // complete tick ops
    WithOpGraphAndMutJob(&AutoTick);
  }
}

}  // namespace oneflow
