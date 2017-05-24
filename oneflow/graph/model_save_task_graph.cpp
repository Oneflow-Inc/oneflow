#include "graph/model_save_task_graph.h"

namespace oneflow {

MdSaveTaskGraph::MdSaveTaskGraph(
    const std::string& name,
    const ChainNode* update_chain,
    const HashMap<uint64_t, CompTaskNode*>& parallel_id2updt_task,
    ParallelPolicy data_chain_policy,
    const std::string& dot_path_prefix) {
  mut_name() = name;
  data_chain_policy_ = data_chain_policy;
  parallel_id2updt_task_ = parallel_id2updt_task;
  BuildTaskGraph(update_chain, dot_path_prefix);
  BuildExecAndEnrollLbn2Regsts();
}

void MdSaveTaskGraph::BuildTaskGraph(const ChainNode* update_chain,
                                     const std::string& dot_path_prefix) {
  auto chain_gph = of_make_unique<ChainGraph> ();
  ChainNode* save_chain = chain_gph->NewNode();
  save_chain->mut_parallel_desc() = update_chain->parallel_desc();
  //
  chain_gph->UpdateSourceAndSink();
  chain_gph->ToDotFile(dot_path_prefix + "chain_graph.dot");
  BuildFromChainGph(std::move(chain_gph), false, dot_path_prefix);
}

} // namespace oneflow
