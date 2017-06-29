#include "oneflow/core/graph/model_diff_accumulate_task_graph.h"
#include "oneflow/core/graph/model_diff_accumulate_comp_task_node.h"

namespace oneflow {

MdDiffAccTaskGraph::MdDiffAccTaskGraph(
    const std::string& name,
    const ChainNode* data_chain,
    const std::vector<CompTaskNode*>& sorted_fw_comptasks4data_chain) {
  mut_name() = name;
  BuildTaskGraph(data_chain);
  for (CompTaskNode* fw_task : sorted_fw_comptasks4data_chain) {
    CHECK(parallel_id2fw_task_.emplace(fw_task->parallel_id(), fw_task).second);
  }
  BuildExecAndEnrollLbn2Regsts();
}

void MdDiffAccTaskGraph::BuildTaskGraph(const ChainNode* data_chain) {
  // Construct ModelDiffAccOp
  OperatorConf op_conf;
  op_conf.set_name("model_diff_acc_" + NewUniqueId());
  op_conf.mutable_model_diff_acc_conf();
  auto model_diff_acc_op = OpMgr::Singleton().ConstructOp(op_conf);
  // ModelDiffAccChain
  auto chain_gph = of_make_unique<ChainGraph> ();
  ChainNode* diff_acc_chain = chain_gph->NewNode();
  diff_acc_chain->mut_op_vec() = {model_diff_acc_op};
  auto parallel_desc4diff_acc =
      new ParallelDesc(*(data_chain->parallel_desc()));
  parallel_desc4diff_acc->mut_policy() = kModelParallel;
  diff_acc_chain->mut_parallel_desc().reset(parallel_desc4diff_acc);
  // FakerChain
  if (data_chain->parallel_desc()->policy() == kDataParallel) {
    ChainNode* faker_chain = chain_gph->NewNode();
    faker_chain->mut_op_vec().clear();
    auto parallel_desc4faker = new ParallelDesc(*(data_chain->parallel_desc()));
    parallel_desc4faker->mut_policy() = kFakerMdUpdt;
    faker_chain->mut_parallel_desc().reset(parallel_desc4faker);
    faker_chain->mut_output_lbns() = {kBaledBlobName};
    diff_acc_chain->mut_input_lbns() = {kBaledBlobName};
    Connect(faker_chain, chain_gph->NewEdge(), diff_acc_chain);
  }
  //
  chain_gph->UpdateSourceAndSink();
  chain_gph->ToDotWithAutoFilePath();
  BuildFromChainGph<MdDiffAccCompTaskNode>(std::move(chain_gph), false);
}

} // namespace oneflow
