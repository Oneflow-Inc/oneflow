#include "graph/task_graph_manager.h"

namespace oneflow {

void TaskGraphMgr::BuildGraphs() {
  ordered_task_gphs_.clear();
  // data graph
  LOG(INFO) << "Build DataTaskGraph...";
  auto data_task_gph = new DataTaskGraph(
        "data",
        JobDesc::Singleton().train_dlnet_conf(),
        JobDesc::Singleton().strategy(),
        true);
  ordered_task_gphs_.emplace_back(data_task_gph);
  // construct data_chain2sorted_bp_comp_tasks
  HashMap<const ChainNode*, std::vector<CompTaskNode*>>
      data_chain2sorted_bp_comp_tasks;
  for (const auto& node : data_task_gph->nodes()) {
    auto bp_node = dynamic_cast<CompTaskNode*>(node.get());
    if (bp_node == nullptr || bp_node->IsFwNode()) { continue; }
    data_chain2sorted_bp_comp_tasks[bp_node->chain_node()].push_back(bp_node);
  }
  for (auto& pair : data_chain2sorted_bp_comp_tasks) {
    SortByParallelId(&(pair.second));
  }
  // model graph
  for (const auto& pair : data_chain2sorted_bp_comp_tasks) {
    std::string chain_tag = pair.first->op_vec().front()->op_name();
    str_replace(&chain_tag, '/', '_');
    const std::string dot_path_prefix = DotDir() + "/model/" + chain_tag + "_";
    ParallelPolicy policy = pair.first->parallel_desc()->policy();
    // model update
    LOG(INFO) << "Build MdUpdtTaskGraph... for " << chain_tag;
    auto updt_gph = new MdUpdtTaskGraph(
        "md_updt_" + chain_tag,
        pair.first, pair.second, dot_path_prefix + "model_update_");
    ChainNode* updt_chain = updt_gph->chain_gph()->SoleSinkNode();
    auto sorted_updt_tasks = updt_gph->SortedCompTasksInChain(updt_chain);
    HashMap<uint64_t, CompTaskNode*> parallel_id2updt_task;
    for (CompTaskNode* update_task : sorted_updt_tasks) {
      CHECK(parallel_id2updt_task.emplace(
            update_task->parallel_id(), update_task).second);
    }
    // model load save
    LOG(INFO) << "Build MdLoadTaskGraph... for " << chain_tag;
    auto load_gph = new MdLoadTaskGraph(
        "md_load_" + chain_tag,
        updt_chain, parallel_id2updt_task, policy,
        dot_path_prefix + "model_load_");
    LOG(INFO) << "Build MdSaveTaskGraph... for " << chain_tag;
    auto save_gph = new MdSaveTaskGraph(
        "md_save_" + chain_tag,
        updt_chain, parallel_id2updt_task, policy,
        dot_path_prefix + "model_save_");
    ordered_task_gphs_.emplace_back(updt_gph);
    ordered_task_gphs_.emplace_back(load_gph);
    ordered_task_gphs_.emplace_back(save_gph);
  }
  // all exec_graph 2 dot
  for (const auto& task_gph : ordered_task_gphs_) {
    for (const auto& task_node : task_gph->nodes()) {
      std::string file_path = DotDir() + "/exec/";
      file_path += task_node->node_id_str() + ".dot";
      task_node->exec_gph().ToDotFile(file_path);
    }
  }
}

void TaskGraphMgr::InferShape4Regsts() {
  for (auto& task_gph : ordered_task_gphs_) {
    LOG(INFO) << "InferShape... for " << task_gph->name();
    task_gph->InferShapeOfBlobsInProducedRegsts();
  }
}

void TaskGraphMgr::AllTaskNodesToProto(PbRpf<TaskProto>* ret) {
  ret->Clear();
  for (const auto& task_gph : ordered_task_gphs_) {
    for (const auto& task_node : task_gph->nodes()) {
      task_node->ToProto(ret->Add());
    }
  }
}

} // namespace oneflow
