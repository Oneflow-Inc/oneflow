#include "oneflow/core/graph/model_update_task_graph.h"
#include "oneflow/core/graph/model_update_comp_task_node.h"

namespace oneflow {

MdUpdtTaskGraph::MdUpdtTaskGraph(
    const std::string& name, uint32_t random_seed, const ChainNode* data_chain,
    const std::vector<CompTaskNode*>& sorted_diff_acc_tasks,
    const std::vector<CompTaskNode*>& sorted_fw_comptasks4data_chain) {
  mut_name() = name;
  for (int i = 0; i < sorted_diff_acc_tasks.size(); i++) {
    CHECK(parallel_id2diff_acc_task_.emplace(i, sorted_diff_acc_tasks.at(i))
              .second);
  }
  for (int i = 0; i < sorted_fw_comptasks4data_chain.size(); i++) {
    CHECK(parallel_id2fw_task_.emplace(i, sorted_fw_comptasks4data_chain.at(i))
              .second);
  }
  BuildTaskGraph(random_seed, data_chain);
  BuildExecAndEnrollLbn2Regsts();
}

void MdUpdtTaskGraph::BuildTaskGraph(uint32_t random_seed,
                                     const ChainNode* data_chain) {
  auto chain_gph = of_make_unique<ChainGraph>();

  ChainNode* updt_chain = chain_gph->NewNode();
  updt_chain->mut_parallel_desc().reset(
      new ParallelDesc((*(data_chain->parallel_desc()))));
  updt_chain->mut_input_lbns() = {kPackedBlobName};
  updt_chain->mut_op_vec() = {OpMgr::Singleton()->ModelUpdateOp()};

  if (data_chain->parallel_desc()->policy() == kDataParallel) {
    ChainNode* faker_chain = chain_gph->NewNode();
    faker_chain->mut_op_vec().clear();
    auto parallel_desc4faker =
        new ParallelDesc((*(data_chain->parallel_desc())));
    parallel_desc4faker->mut_policy() = kFakerMdUpdt;
    faker_chain->mut_parallel_desc().reset(parallel_desc4faker);
    faker_chain->mut_output_lbns() = {kPackedBlobName};
    Connect(faker_chain, chain_gph->NewEdge(), updt_chain);
  }

  chain_gph->UpdateSourceAndSink();
  chain_gph->ToDotWithAutoFilePath();
  BuildFromChainGph<MdUpdtCompTaskNode>(std::move(chain_gph), false);

  LOG(INFO) << "------------";
  ForEachNode([this, random_seed](TaskNode* node) {
    auto model_updt_comp_task_node = dynamic_cast<MdUpdtCompTaskNode*>(node);
    if (model_updt_comp_task_node == nullptr) { return; }
    auto parallel_id = model_updt_comp_task_node->parallel_id();
    LOG(INFO) << "parallel_id: " << parallel_id;
    model_updt_comp_task_node->set_fw_task(GetFwTask(parallel_id));
    model_updt_comp_task_node->set_diff_acc_task(GetDiffAccTask(parallel_id));
    ParallelPolicy this_policy =
        GetFwTask(parallel_id)->chain_node()->parallel_desc()->policy();
    if (this_policy == kDataParallel) {
      model_updt_comp_task_node->set_random_seed(random_seed);
    } else if (this_policy == kModelParallel) {
      model_updt_comp_task_node->set_random_seed(NewRandomSeed());
    } else {
      UNEXPECTED_RUN();
    }
  });
}

}  // namespace oneflow
