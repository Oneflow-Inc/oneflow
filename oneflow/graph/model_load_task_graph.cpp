#include "graph/model_load_task_graph.h"
#include "job/job_desc.h"

namespace oneflow {

namespace {

void SetModelLoadChain(ChainNode* model_load_chain) {
  // model load op
  OperatorConf op_conf;
  op_conf.set_name("");
  op_conf.mutable_model_load_op_conf();
  model_load_chain->mut_op_vec() = {ConstructOpFromPbConf(op_conf)};
  // model load parallel_conf
  ParallelConf pr_conf;
  pr_conf.set_policy(kDataParallel);
  pr_conf.add_devices(JobDesc::Singleton().MdLoadMachine() + "/disk_loader");
  model_load_chain->mut_parallel_desc().reset(new ParallelDesc(pr_conf));
  // output
  model_load_chain->mut_output_lbns() = {RegstDesc::kAllLbn};
}

} // namespace

MdLoadTaskGraph::MdLoadTaskGraph(const MdUpdtTaskGraph* md_updt_gph) {
  BuildTaskGraph(md_updt_gph->chain_gph()->SoleLastNode());
  InitFaker2Mccoy(md_updt_gph);
  BuildExecAndProducedRegsts();
}

void MdLoadTaskGraph::BuildTaskGraph(const ChainNode* update_chain) {
  auto chain_gph = make_unique<ChainGraph> ();
  ChainNode* load_chain = chain_gph->NewFinalNode();
  SetModelLoadChain(load_chain);
  ChainNode* faker_chain = chain_gph->NewFinalNode();
  faker_chain->mut_op_vec() = {};
  faker_chain->mut_parallel_desc() = update_chain->parallel_desc();
  faker_chain->mut_input_lbns() = {RegstDesc::kAllLbn};
  Connect(load_chain, chain_gph->NewFinalEdge(), faker_chain);
  chain_gph->UpdateSourceAndSink();
  BuildFromChainGph(std::move(chain_gph), false);
}

void MdLoadTaskGraph::InitFaker2Mccoy(const MdUpdtTaskGraph* md_updt_gph) {
  std::vector<CompTaskNode*> sorted_update_tasks;
  for (const auto& node : md_updt_gph->nodes()) {
    auto comp_node = dynamic_cast<CompTaskNode*> (node.get());
    if (comp_node == nullptr || comp_node->IsFaker()) { continue; }
    sorted_update_tasks.push_back(comp_node);
  }
  SortByParallelId(&sorted_update_tasks);
  std::vector<CompTaskNode*> sorted_faker_tasks;
  for (const auto& node : nodes()) {
    auto comp_node = dynamic_cast<CompTaskNode*> (node.get());
    if (comp_node == nullptr || !comp_node->IsFaker()) { continue; }
    sorted_faker_tasks.push_back(comp_node);
  }
  SortByParallelId(&sorted_faker_tasks);
  CHECK_EQ(sorted_update_tasks.size(), sorted_faker_tasks.size());
  for (size_t i = 0; i < sorted_update_tasks.size(); ++i) {
    EnrollFakerMccoy(sorted_faker_tasks[i], sorted_update_tasks[i]);
  }
}

} // namespace oneflow
