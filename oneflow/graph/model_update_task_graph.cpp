#include "graph/model_update_task_graph.h"
#include "operator/operator_manager.h"

namespace oneflow {

MdUpdtTaskGraph::MdUpdtTaskGraph(
    const ChainNode* data_chain,
    const std::vector<CompTaskNode*>& sorted_bp_comptasks4data_chain,
    const std::string& dot_path_prefix) {
  BuildTaskGraph(data_chain, dot_path_prefix);
  for (CompTaskNode* bp_task : sorted_bp_comptasks4data_chain) {
    CHECK(parallel_id2bp_task_.emplace(bp_task->parallel_id(), bp_task).second);
  }
  BuildExecAndEnrollLbn2Regsts();
}

void MdUpdtTaskGraph::BuildTaskGraph(const ChainNode* data_chain,
                                     const std::string& dot_path_prefix) {
  // Construct ModelUpdateOp
  OperatorConf op_conf;
  op_conf.set_name("model_update_" + NewUniqueId());
  op_conf.mutable_model_update_conf();
  auto model_update_op = OpMgr::Singleton().ConstructOp(op_conf);
  // ModelUpdateChain
  auto chain_gph = of_make_unique<ChainGraph> ();
  ChainNode* updt_chain = chain_gph->NewNode();
  updt_chain->mut_op_vec() = {model_update_op};
  auto parallel_desc4updt = new ParallelDesc(*(data_chain->parallel_desc()));
  parallel_desc4updt->mut_policy() = kModelParallel;
  updt_chain->mut_parallel_desc().reset(parallel_desc4updt);
  // FakerChain
  if (data_chain->parallel_desc()->policy() == kDataParallel) {
    ChainNode* faker_chain = chain_gph->NewNode();
    faker_chain->mut_op_vec().clear();
    faker_chain->mut_parallel_desc() = data_chain->parallel_desc();
    faker_chain->mut_output_lbns() = {RegstDesc::kAllLbn};
    updt_chain->mut_input_lbns() = {RegstDesc::kAllLbn};
    Connect(faker_chain, chain_gph->NewEdge(), updt_chain);
  }
  //
  chain_gph->UpdateSourceAndSink();
  chain_gph->ToDotFile(dot_path_prefix + "chain_graph.dot");
  BuildFromChainGph(std::move(chain_gph), false, dot_path_prefix);
}

} // namespace oneflow
