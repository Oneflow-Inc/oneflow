#include "path/model_update_path.h"

namespace oneflow {

namespace {

void AddConsumedModelLbnRegst4FwExecGraph(
    const ExecGraph& exec_gph,
    RegstDesc* model_regst) {
  for (const auto& exec_node : exec_gph.nodes()) {
    for (const std::string& mbn : exec_node->op()->model_bns()) {
      std::string lbn = exec_node->op()->mbn2lbn(mbn);
      exec_node->AddConsumedLbnRegstPair(lbn, model_regst);
    }
  }
}

}

void ModelUpdatePath::Build(
    const ChainNode* data_chain,
    const std::vector<CompTaskNode*>& sorted_bp_comptasks4data_chain) {
  set_data_chain(data_chain);
  BuildTaskGraph(data_chain);
  HashMap<int32_t, CompTaskNode*> parallel_id2update_node;
  InitFaker2MccoyMapAndParallelIdUpdateMap(sorted_bp_comptasks4data_chain,
                                           &parallel_id2update_node);
  BuildExecAndProducedRegstsAndSubscribeInPath();
  if (faker2mccoy().empty()) {
    SubscribeCrossPathWithoutFaker(sorted_bp_comptasks4data_chain,
                                   parallel_id2update_node);
  } else {
    SubscribeCrossPathWithFaker(parallel_id2update_node);
  }
}

void ModelUpdatePath::BuildTaskGraph(const ChainNode* data_chain) {
  // Construct ModelUpdateOp
  OperatorConf op_conf;
  op_conf.set_name("model_update_" + data_chain->ConcatedOpsName());
  op_conf.mutable_model_update_op_conf();
  auto model_update_op = ConstructOpFromPbConf(op_conf);
  // Useful vars
  auto parallel_desc4data_chain = data_chain->parallel_desc();
  std::unique_ptr<ChainGraph> chain_gph(new ChainGraph);
  // ModelUpdateChain
  ChainNode* model_update_chain = chain_gph->NewFinalNode();
  model_update_chain->mut_op_vec() = {model_update_op};
  auto parallel_desc4model_update = new ParallelDesc(*parallel_desc4data_chain);
  parallel_desc4model_update->mut_policy() = kModelParallel;
  model_update_chain->mut_parallel_desc().reset(parallel_desc4model_update);
  // FakerChain
  if (parallel_desc4data_chain->policy() == kDataParallel) {
    ChainNode* faker_chain = chain_gph->NewFinalNode();
    faker_chain->mut_op_vec().clear();
    faker_chain->mut_parallel_desc() = parallel_desc4data_chain;
    faker_chain->mut_output_lbns() = {ContigRegstDesc::kAllLbn};
    model_update_chain->mut_input_lbns() = {ContigRegstDesc::kAllLbn};
    Connect(faker_chain, chain_gph->NewFinalEdge(), model_update_chain);
  }
  // 
  mut_task_gph().reset(new TaskGraph(std::move(chain_gph), false));
}

void ModelUpdatePath::InitFaker2MccoyMapAndParallelIdUpdateMap(
    const std::vector<CompTaskNode*>& sorted_bp_comptasks4data_chain,
    HashMap<int32_t, CompTaskNode*>* parallel_id2update_node) {
  std::vector<CompTaskNode*> comptasks4faker_chain;
  for (const std::unique_ptr<TaskNode>& node : task_gph()->nodes()) {
    CompTaskNode* comp_node = dynamic_cast<CompTaskNode*> (node.get());
    if (!comp_node) { continue; }
    if (comp_node->IsFaker()) {
      comptasks4faker_chain.push_back(comp_node);
    } else {
      parallel_id2update_node->emplace(comp_node->parallel_id(), comp_node);
    }
  }
  SortByParallelId(&comptasks4faker_chain);
  CHECK_EQ(comptasks4faker_chain.size(), sorted_bp_comptasks4data_chain.size());
  for (size_t i = 0; i < comptasks4faker_chain.size(); ++i) {
    AddFakerMccoyPair(comptasks4faker_chain[i],
                      sorted_bp_comptasks4data_chain[i]);
  }
}

void ModelUpdatePath::SubscribeCrossPathWithoutFaker(
    const std::vector<CompTaskNode*>& sorted_bp_comptasks4data_chain,
    const HashMap<int32_t, CompTaskNode*>& parallel_id2update_node) {
  for (CompTaskNode* bp_task_node : sorted_bp_comptasks4data_chain) {
    // Useful Vars
    int32_t parallel_id = bp_task_node->parallel_id();
    CompTaskNode* update_task_node = parallel_id2update_node.at(parallel_id);
    TaskNode* fw_task_node = bp_task_node->GetFwNode();
    ExecNode* update_exec_node = update_task_node->exec_gph().SoleNode();
    RegstDesc* model_diff_regst = bp_task_node->GetProducedRegstDesc("model_diff");
    RegstDesc* model_regst = update_task_node->GetProducedRegstDesc("model");
    // update_node Subscribe ModelDiffRegst
    update_task_node->Subscribe(model_diff_regst);
    update_exec_node->AddConsumedLbnRegstPair(ContigRegstDesc::kAllLbn,
                                             model_diff_regst);
    // fw_node Subscribe ModelRegst
    fw_task_node->Subscribe(model_regst);
    AddConsumedModelLbnRegst4FwExecGraph(fw_task_node->exec_gph(), model_regst);
  }
}

void ModelUpdatePath::SubscribeCrossPathWithFaker(
    const HashMap<int32_t, CompTaskNode*>& parallel_id2update_node) {
  for (const auto& pair : faker2mccoy()) {
    int32_t parallel_id = pair.first->parallel_id();
    CompTaskNode* update_node = parallel_id2update_node.at(parallel_id);
    TaskNode* fw_comp_node = pair.second->GetFwNode();
    RegstDesc* model_regst = update_node->GetProducedRegstDesc("model");
    fw_comp_node->Subscribe(model_regst);
    AddConsumedModelLbnRegst4FwExecGraph(fw_comp_node->exec_gph(), model_regst);
  }
}

} // namespace oneflow
