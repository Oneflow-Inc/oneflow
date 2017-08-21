#include "oneflow/core/graph/model_update_task_graph.h"
#include "oneflow/core/graph/model_update_comp_task_node.h"

namespace oneflow {

MdUpdtTaskGraph::MdUpdtTaskGraph(const std::string& name, CompTaskNode* fw_task,
                                 CompTaskNode* diff_acc_task,
                                 uint32_t random_seed) {
  mut_name() = name;
  fw_task_ = fw_task;
  diff_acc_task_ = diff_acc_task;
  BuildTaskGraph(random_seed);
  BuildExecAndEnrollLbn2Regsts();
}

void MdUpdtTaskGraph::BuildTaskGraph(uint32_t random_seed) {
  auto chain_gph = of_make_unique<ChainGraph>();

  ChainNode* updt_chain = chain_gph->NewNode();
  ParallelConf updt_pr_conf;
  updt_pr_conf.set_policy(kDataParallel);
  updt_pr_conf.add_device_name(fw_task_->device_name());
  updt_chain->mut_parallel_desc().reset(new ParallelDesc(updt_pr_conf));
  updt_chain->mut_input_lbns() = {kPackedBlobName};
  updt_chain->mut_op_vec() = {OpMgr::Singleton()->ModelUpdateOp()};
  chain_gph->UpdateSourceAndSink();
  chain_gph->ToDotWithAutoFilePath();
  BuildFromChainGph<MdUpdtCompTaskNode>(std::move(chain_gph), false);

  ForEachNode([this, random_seed](TaskNode* node) {
    auto model_updt_comp_task_node = dynamic_cast<MdUpdtCompTaskNode*>(node);
    if (model_updt_comp_task_node == nullptr) { return; }
    model_updt_comp_task_node->set_fw_task(fw_task_);
    ParallelPolicy this_policy =
        fw_task_->chain_node()->parallel_desc()->policy();
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
