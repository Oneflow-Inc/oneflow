#include "path/model_update_path.h"

namespace oneflow {

void ModelUpdatePath::Build(const ChainNode* chain_in_data_path) {
  BuildTaskGraph(chain_in_data_path);
  TODO();
  // BuildFakerMap
  // ProducedRegister SubscribeRegister
  // data_path subscribe model_register
}

void ModelUpdatePath::BuildTaskGraph(const ChainNode* chain_in_data_path) {
  // Construct ModelUpdateOp
  OperatorConf op_conf;
  op_conf.set_name("model_update_" + chain_in_data_path->ConcatedOpsName());
  op_conf.mutable_model_update_op_conf();
  std::shared_ptr<Operator> model_update_op = ConstructOpFromPbConf(op_conf);
  // Useful vars
  std::shared_ptr<const ParallelDesc> parallel_desc_data = chain_in_data_path->parallel_desc();
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

} // namespace oneflow
