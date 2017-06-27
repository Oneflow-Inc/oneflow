#include "oneflow/core/graph/model_update_task_graph.h"
#include "oneflow/core/graph/model_update_comp_task_node.h"

namespace oneflow {

MdUpdtTaskGraph::MdUpdtTaskGraph(const std::string& name,
                                 CompTaskNode* fw_task,
                                 CompTaskNode* diff_acc_task) {
  mut_name() = name;
  fw_task_ = fw_task;
  diff_acc_task_ = diff_acc_task;
  BuildTaskGraph();
  BuildExecAndEnrollLbn2Regsts();
}

void MdUpdtTaskGraph::BuildTaskGraph() {
  auto chain_gph = of_make_unique<ChainGraph> ();
  OperatorConf op_conf;
  op_conf.set_name("model_update_"+ NewUniqueId());
  op_conf.mutable_model_update_conf();
  auto model_updt_op = OpMgr::Singleton().ConstructOp(op_conf);

  ChainNode* updt_chain = chain_gph->NewNode();
  ParallelConf updt_pr_conf;
  updt_pr_conf.set_policy(kDataParallel);
  updt_pr_conf.mutable_device_set()->add_device_name(fw_task_->device_name());
  updt_chain->mut_parallel_desc().reset(new ParallelDesc(updt_pr_conf));
  updt_chain->mut_input_lbns() = {kBaledBlobName};
  updt_chain->mut_op_vec() = {model_updt_op};
  chain_gph->UpdateSourceAndSink();
  chain_gph->ToDotWithAutoFilePath();
  BuildFromChainGph<MdUpdtCompTaskNode>(std::move(chain_gph), false);

  ForEachNode([this](TaskNode* node) {
    auto model_updt_comp_task_node = dynamic_cast<MdUpdtCompTaskNode*>(node);
    if (model_updt_comp_task_node != nullptr) {
      model_updt_comp_task_node->set_fw_task(fw_task_);
    }
  });
}

} // namespace oneflow
