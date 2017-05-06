#include "graph/model_load_task_graph.h"
#include "job/job_desc.h"

namespace oneflow {

namespace {

void SetModelLoadChain(ChainNode* model_load_chain) {
  // model load op
  OperatorConf op_conf;
  op_conf.set_name("model_load_" + NewUniqueId());
  op_conf.mutable_model_load_conf();
  model_load_chain->mut_op_vec() = {OpMgr::Singleton().ConstructOp(op_conf)};
  // model load parallel_conf
  ParallelConf pr_conf;
  pr_conf.set_policy(kDataParallel);
  pr_conf.mutable_device_set()->add_device_name(
      JobDesc::Singleton().md_load_machine() + ":disk");
  model_load_chain->mut_parallel_desc().reset(new ParallelDesc(pr_conf));
  // output
  model_load_chain->mut_output_lbns() = {RegstDesc::kAllLbn};
}

} // namespace

MdLoadTaskGraph::MdLoadTaskGraph(
    const ChainNode* update_chain,
    const HashMap<uint64_t, CompTaskNode*>& parallel_id2updt_task,
    ParallelPolicy policy,
    const std::string& dot_path_prefix) {
  mut_policy() = policy;
  BuildTaskGraph(update_chain, dot_path_prefix);
  mut_parallel_id2updt_task() = parallel_id2updt_task;
  BuildExecAndEnrollLbn2Regsts();
}

void MdLoadTaskGraph::BuildTaskGraph(const ChainNode* update_chain,
                                     const std::string& dot_path_prefix) {
  auto chain_gph = of_make_unique<ChainGraph> ();
  ChainNode* load_chain = chain_gph->NewNode();
  SetModelLoadChain(load_chain);
  ChainNode* faker_chain = chain_gph->NewNode();
  faker_chain->mut_op_vec() = {};
  faker_chain->mut_parallel_desc() = update_chain->parallel_desc();
  faker_chain->mut_input_lbns() = {RegstDesc::kAllLbn};
  Connect(load_chain, chain_gph->NewEdge(), faker_chain);
  chain_gph->UpdateSourceAndSink();
  chain_gph->ToDotFile(dot_path_prefix + "chain_graph.dot");
  BuildFromChainGph(std::move(chain_gph), false, dot_path_prefix);
}

} // namespace oneflow
