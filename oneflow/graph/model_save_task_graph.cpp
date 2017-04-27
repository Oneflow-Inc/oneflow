#include "graph/model_save_task_graph.h"

namespace oneflow {

namespace {

void SetModelSaveChain(ChainNode* model_save_chain) {
  // model save op
  OperatorConf op_conf;
  op_conf.set_name("");
  op_conf.mutable_model_save_conf();
  model_save_chain->mut_op_vec() = {OpMgr::Singleton().ConstructOp(op_conf)};
  // model save parallel_conf
  ParallelConf pr_conf;
  pr_conf.set_policy(kDataParallel);
  pr_conf.mutable_device_set()->add_device_name(JobDesc::Singleton().md_save_machine() + "/disk");
  model_save_chain->mut_parallel_desc().reset(new ParallelDesc(pr_conf));
  // output
  model_save_chain->mut_input_lbns() = {RegstDesc::kAllLbn};
}

} // namespace

MdSaveTaskGraph::MdSaveTaskGraph(
    const ChainNode* update_chain,
    const std::vector<CompTaskNode*>& sorted_updt_tasks) {
  BuildTaskGraph(update_chain);
  InitFaker2Mccoy(sorted_updt_tasks);
  BuildExecAndProducedRegsts();
}

void MdSaveTaskGraph::BuildTaskGraph(const ChainNode* update_chain) {
  auto chain_gph = of_make_unique<ChainGraph> ();
  // faker
  ChainNode* faker_chain = chain_gph->NewNode();
  faker_chain->mut_parallel_desc() = update_chain->parallel_desc();
  faker_chain->mut_output_lbns() = {RegstDesc::kAllLbn};
  // save
  ChainNode* save_chain = chain_gph->NewNode();
  SetModelSaveChain(save_chain);
  // Connect
  Connect(faker_chain, chain_gph->NewEdge(), save_chain);
  chain_gph->UpdateSourceAndSink();
  BuildFromChainGph(std::move(chain_gph), false);
}

void MdSaveTaskGraph::InitFaker2Mccoy(
    const std::vector<CompTaskNode*>& sorted_updt_tasks) {
  auto sorted_faker_tasks = SortedCompTasksInChain(chain_gph()->SoleSourceNode());
  CHECK_EQ(sorted_updt_tasks.size(), sorted_faker_tasks.size());
  for (size_t i = 0; i < sorted_updt_tasks.size(); ++i) {
    EnrollFakerMccoy(sorted_faker_tasks[i], sorted_updt_tasks[i]);
  }
}

} // namespace oneflow
