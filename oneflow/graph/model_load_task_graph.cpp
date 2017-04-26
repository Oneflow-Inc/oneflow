#include "graph/model_load_task_graph.h"
#include "job/job_desc.h"

namespace oneflow {

namespace {

void SetModelLoadChain(ChainNode* model_load_chain) {
  // model load op
  OperatorConf op_conf;
  op_conf.set_name("");
  op_conf.mutable_model_load_conf();
  model_load_chain->mut_op_vec() = {ConstructOpFromPbConf(op_conf)};
  // model load parallel_conf
  ParallelConf pr_conf;
  pr_conf.set_policy(kDataParallel);
  pr_conf.mutable_device_set()->add_device_name(JobDesc::Singleton().md_load_machine() + "/disk");
  model_load_chain->mut_parallel_desc().reset(new ParallelDesc(pr_conf));
  // output
  model_load_chain->mut_output_lbns() = {RegstDesc::kAllLbn};
}

} // namespace

MdLoadTaskGraph::MdLoadTaskGraph(
    const ChainNode* update_chain,
    const std::vector<CompTaskNode*>& sorted_update_tasks) {
  BuildTaskGraph(update_chain);
  InitFaker2Mccoy(sorted_update_tasks);
  BuildExecAndProducedRegsts();
}

void MdLoadTaskGraph::BuildTaskGraph(const ChainNode* update_chain) {
  auto chain_gph = of_make_unique<ChainGraph> ();
  ChainNode* load_chain = chain_gph->NewNode();
  SetModelLoadChain(load_chain);
  ChainNode* faker_chain = chain_gph->NewNode();
  faker_chain->mut_op_vec() = {};
  faker_chain->mut_parallel_desc() = update_chain->parallel_desc();
  faker_chain->mut_input_lbns() = {RegstDesc::kAllLbn};
  Connect(load_chain, chain_gph->NewEdge(), faker_chain);
  chain_gph->UpdateSourceAndSink();
  BuildFromChainGph(std::move(chain_gph), false);
}

void MdLoadTaskGraph::InitFaker2Mccoy(
    const std::vector<CompTaskNode*>& sorted_update_tasks) {
  auto sorted_faker_tasks = SortedCompTasksInChain(chain_gph()->SoleSinkNode());
  CHECK_EQ(sorted_update_tasks.size(), sorted_faker_tasks.size());
  for (size_t i = 0; i < sorted_update_tasks.size(); ++i) {
    EnrollFakerMccoy(sorted_faker_tasks[i], sorted_update_tasks[i]);
  }
}

} // namespace oneflow
