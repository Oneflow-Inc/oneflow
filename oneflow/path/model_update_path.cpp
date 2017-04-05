#include "path/model_update_path.h"

namespace oneflow {

void ModelUpdatePath::Build(
    const ChainNode* data_chain,
    const std::vector<CompTaskNode*>& sorted_comptasks4data_chain) {
  BuildTaskGraph(data_chain);
  InitFaker2DataMap(sorted_comptasks4data_chain);
  TODO();
  // BuildFakerMap
  // ProducedRegister SubscribeRegister
  // data_path subscribe model_register
}

void ModelUpdatePath::BuildTaskGraph(const ChainNode* data_chain) {
  // Construct ModelUpdateOp
  OperatorConf op_conf;
  op_conf.set_name("model_update_" + data_chain->ConcatedOpsName());
  op_conf.mutable_model_update_op_conf();
  std::shared_ptr<Operator> model_update_op = ConstructOpFromPbConf(op_conf);
  // Useful vars
  std::shared_ptr<const ParallelDesc> parallel_desc_data = data_chain->parallel_desc();
  std::unique_ptr<ChainGraph> chain_gph(new ChainGraph);
  // ModelUpdateChain
  ChainNode* model_update_chain = chain_gph->NewFinalNode();
  model_update_chain->mut_op_vec() = {model_update_op};
  auto parallel_desc_model_update = new ParallelDesc(*parallel_desc_data);
  parallel_desc_model_update->mut_policy() = kModelParallel;
  model_update_chain->mut_parallel_desc().reset(parallel_desc_model_update);
  // FakerChain
  if (parallel_desc_data->policy() == kDataParallel) {
    ChainNode* faker_chain = chain_gph->NewFinalNode();
    faker_chain->mut_op_vec().clear();
    faker_chain->mut_parallel_desc() = parallel_desc_data;
    Connect(faker_chain, chain_gph->NewFinalEdge(), model_update_chain);
  }
  // 
  mut_task_graph().reset(new TaskGraph(std::move(chain_gph), false));
}

void ModelUpdatePath::InitFaker2DataMap(
    const std::vector<CompTaskNode*>& sorted_comptasks4data_chain) {
  std::vector<CompTaskNode*> comptasks4faker_chain;
  for (const std::unique_ptr<TaskNode>& node : task_graph()->nodes()) {
    CompTaskNode* comp_node = nullptr;
    if (comp_node = dynamic_cast<CompTaskNode*>(node.get()), !comp_node) {
      continue;
    }
    if (comp_node->chain_node()->op_vec().empty()) {
      comptasks4faker_chain.push_back(comp_node);
    }
  }
  SortByParallelId(&comptasks4faker_chain);
  CHECK_EQ(comptasks4faker_chain.size(), sorted_comptasks4data_chain.size());
  for (size_t i = 0; i < comptasks4faker_chain.size(); ++i) {
    CHECK(faker2data.emplace(comptasks4faker_chain[i],
                             sorted_comptasks4data_chain[i]).second);
  }
}

} // namespace oneflow
