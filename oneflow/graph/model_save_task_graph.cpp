#include "graph/model_save_task_graph.h"

namespace oneflow {

namespace {

void SetModelSaveChain(ChainNode* model_save_chain) {
  // model save op
  OperatorConf op_conf;
  op_conf.set_name("model_save_" + NewUniqueId());
  op_conf.mutable_model_save_conf();
  model_save_chain->mut_op_vec() = {OpMgr::Singleton().ConstructOp(op_conf)};
  // model save parallel_conf
  ParallelConf pr_conf;
  pr_conf.set_policy(kDataParallel);
  pr_conf.mutable_device_set()->add_device_name(
      JobDesc::Singleton().md_save_machine() + ":disk");
  model_save_chain->mut_parallel_desc().reset(new ParallelDesc(pr_conf));
  // output
  model_save_chain->mut_input_lbns() = {RegstDesc::kAllLbn};
}

} // namespace

MdSaveTaskGraph::MdSaveTaskGraph(
    const ChainNode* update_chain,
    const HashMap<uint64_t, CompTaskNode*>& parallel_id2updt_task,
    ParallelPolicy policy,
    const std::string& dot_path_prefix) {
  mut_policy() = policy;
  mut_parallel_id2updt_task() = parallel_id2updt_task;
  BuildTaskGraph(update_chain, dot_path_prefix);
  BuildExecAndEnrollLbn2Regsts();
}

void MdSaveTaskGraph::BuildTaskGraph(const ChainNode* update_chain,
                                     const std::string& dot_path_prefix) {
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
  chain_gph->ToDotFile(dot_path_prefix + "chain_graph.dot");
  BuildFromChainGph(std::move(chain_gph), false, dot_path_prefix);
}

} // namespace oneflow
