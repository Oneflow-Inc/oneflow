#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/copy_task_node.h"

namespace oneflow {

TaskGraph::TaskGraph(std::unique_ptr<const ChainGraph>&& chain_gph) {
  chain_gph_ = std::move(chain_gph);
  HashMap<const ChainNode*, std::vector<CompTaskNode*>> chain2sorted_comp_tasks;
  chain_gph_->ForEachNode([&](const ChainNode* chain_node) {
    chain_node->GenSortedCompTaskNodes([&](CompTaskNode* comp_task_node) {
      comp_task_node->FixThrdLocId();
      AddAllocatedNode(comp_task_node);
      chain2sorted_comp_tasks[chain_node].push_back(comp_task_node);
    });
  });
  chain_gph_->ForEachEdge([&](const ChainEdge* chain_edge) {
    BldSubTskGphMthd method = chain_edge->GetMthdForBldSubTskGph();
    (this->*method)(chain_edge->src_node(), chain_edge->dst_node(),
                    chain2sorted_comp_tasks.at(chain_edge->src_node()),
                    chain2sorted_comp_tasks.at(chain_edge->dst_node()));
  });
}

void TaskGraph::BldSubTskGphByNormalBoxing(
    const ChainNode* src_chain, const ChainNode* dst_chain,
    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks) {}

void TaskGraph::BldSubTskGphByAddCloneBoxing(
    const ChainNode* src_chain, const ChainNode* dst_chain,
    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks) {}

void TaskGraph::BldSubTskGphByOneToOne(
    const ChainNode* src_chain, const ChainNode* dst_chain,
    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks) {
  CHECK_EQ(sorted_src_comp_tasks.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(size_t, i, 0, sorted_src_comp_tasks.size()) {
    CompTaskNode* src_comp_task = sorted_src_comp_tasks[i];
    CompTaskNode* dst_comp_task = sorted_dst_comp_tasks[i];

    if (src_comp_task->machine_id() == dst_comp_task->machine_id()) {
      if (src_comp_task->device_type() == dst_comp_task->device_type()) {
        if (src_comp_task->device_type() != DeviceType::kCPU) {
          CHECK_EQ(src_comp_task->thrd_loc_id(), dst_comp_task->thrd_loc_id());
        }
        Connect<TaskNode>(src_comp_task, NewEdge(), dst_comp_task);
      } else {
        CopyHdTaskNode* copy_hd_task = NewNode<CopyHdTaskNode>();
        if (src_comp_task->device_type() == DeviceType::kCPU) {
          TODO();
        } else {
          TODO();
        }
        Connect<TaskNode>(src_comp_task, NewEdge(), copy_hd_task);
        Connect<TaskNode>(copy_hd_task, NewEdge(), dst_comp_task);
      }
    } else {
      TODO();
    }
  }
}

void TaskGraph::BldSubTskGphBySelectOneSourceToSoleSink(
    const ChainNode* src_chain, const ChainNode* dst_chain,
    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks) {}

}  // namespace oneflow
