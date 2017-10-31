#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

TaskGraph::TaskGraph(std::unique_ptr<const ChainGraph>&& chain_gph) {
  chain_gph_ = std::move(chain_gph);
  HashMap<const ChainNode*, std::vector<CompTaskNode*>> chain2comp_tasks;
  chain_gph_->ForEachNode([&](const ChainNode* chain_node) {
    chain_node->GenSortedCompTaskNodes([&](CompTaskNode* comp_task_node) {
      comp_task_node->FixThrdLocId();
      AddAllocatedNode(comp_task_node);
      chain2comp_tasks[chain_node].push_back(comp_task_node);
    });
  });
  chain_gph_->ForEachEdge([&](const ChainEdge* chain_edge) {
    BldSubTskGphMthd method = chain_edge->GetMthdForBldSubTskGph();
    (this->*method)(chain_edge->src_node(), chain_edge->dst_node());
  });
}

void TaskGraph::BldSubTskGphByNormalBoxing(const ChainNode* src,
                                           const ChainNode* dst) {}

void TaskGraph::BldSubTskGphByAddCloneBoxing(const ChainNode* src,
                                             const ChainNode* dst) {}

void TaskGraph::BldSubTskGphByDirectOneToOne(const ChainNode* src,
                                             const ChainNode* dst) {}

void TaskGraph::BldSubTskGphByInDirectOneToOne(const ChainNode* src,
                                               const ChainNode* dst) {}

void TaskGraph::BldSubTskGphBySelectOneSourceToSoleSink(const ChainNode* src,
                                                        const ChainNode* dst) {}

}  // namespace oneflow
