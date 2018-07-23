#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

namespace {

void InitChains(const std::vector<TaskNode*>& ordered_nodes, std::list<Chain>* chain_list,
                Task2ChainItMap* task2chain_it) {
  chain_list->clear();
  task2chain_it->clear();
  for (const auto& task_node : ordered_nodes) {
    chain_list->emplace_back();
    task2chain_it->insert({task_node, --chain_list->end()});
    Chain& cur_chain = chain_list->back();
    cur_chain.nodes = {task_node};
    cur_chain.area_id = task_node->area_id();
    cur_chain.stream_id = task_node->GlobalWorkStreamId();
    for (auto& node : cur_chain.nodes) { cur_chain.ancestors_and_this.set(node->task_uid()); }
    for (auto& node : task_node->ancestors()) {
      cur_chain.ancestors.set(node->task_uid());
      cur_chain.ancestors_and_this.set(node->task_uid());
    }
  }
}

bool DoMerge(std::list<ChainIt>& chains, ChainIt rhs, Task2ChainItMap* task2chain_it) {
  for (auto chains_it = chains.rbegin(); chains_it != chains.rend(); ++chains_it) {
    ChainIt lhs = *chains_it;
    if (lhs->ancestors_and_this == (lhs->ancestors_and_this | rhs->ancestors)) {
      for (TaskNode* node : rhs->nodes) {
        lhs->nodes.push_back(node);
        lhs->ancestors_and_this.set(node->task_uid());
        task2chain_it->at(node) = lhs;
      }
      return true;
    }
  }
  return false;
}

bool TryMerge(
    std::list<Chain>* chain_list, Task2ChainItMap* task2chain_it,
    std::function<bool(std::list<ChainIt>& chains, ChainIt cur_it, Task2ChainItMap* task2chain_it)>
        DoMerge) {
  HashMap<std::pair<int64_t, int64_t>, std::list<ChainIt>> stream_area2chains;
  bool merge_happened = false;
  for (auto cur_chain_it = chain_list->begin(); cur_chain_it != chain_list->end();) {
    std::pair<int64_t, int64_t> stream_area_id = {cur_chain_it->stream_id, cur_chain_it->area_id};
    auto stream_area_it = stream_area2chains.find(stream_area_id);
    if (stream_area_it != stream_area2chains.end()
        && DoMerge(stream_area_it->second, cur_chain_it, task2chain_it)) {
      cur_chain_it = chain_list->erase(cur_chain_it);
      merge_happened = true;
    } else {
      stream_area2chains[stream_area_id].push_back(cur_chain_it);
      ++cur_chain_it;
    }
  }
  return merge_happened;
}

void MergeChains(std::list<Chain>* chain_list, Task2ChainItMap* task2chain_it) {
  while (TryMerge(chain_list, task2chain_it, DoMerge)) {}
}

}  // namespace

std::string ChainNode::VisualStr() const {
  std::stringstream ss;
  ss << "chain_id:" << chain_id_ << "\\n";
  for (auto& task_node : chain_it_->nodes) {
    ss << TaskType_Name(task_node->GetTaskType()) << ":" << task_node->task_id() << "\\n";
  }
  return ss.str();
}

ChainGraph::ChainGraph(const TaskGraph& task_gph) : task_gph_(task_gph) {
  std::vector<TaskNode*> ordered_task_nodes;
  task_gph.AcyclicTopoForEachNode([&](TaskNode* node) { ordered_task_nodes.emplace_back(node); });

  InitChains(ordered_task_nodes, &chain_list_, &task_node2chain_it_);
  MergeChains(&chain_list_, &task_node2chain_it_);

  for (auto chain_it = chain_list_.begin(); chain_it != chain_list_.end(); ++chain_it) {
    ChainNode* chain_node = new ChainNode(chain_it);
    chain_it->chain_node = chain_node;
    AddAllocatedNode(chain_node);
  }

  for (auto& cur_task_node : ordered_task_nodes) {
    auto cur_chain_node = ChainNode4TaskNode(cur_task_node);
    for (auto& task_in_edge : cur_task_node->in_edges()) {
      auto src_task_node = task_in_edge->src_node();
      if (!cur_task_node->ancestors().count(src_task_node)) {
        continue;  // ignore kMdUpdt-{kNormalForward, kNormalBackward} edge
      }
      auto src_chain_node = ChainNode4TaskNode(src_task_node);
      if (cur_chain_node == src_chain_node) continue;
      if (HasChainEdge(src_chain_node, cur_chain_node)) continue;
      Connect(src_chain_node, NewEdge(), cur_chain_node);
    }
  }

  TopoForEachNode([&](ChainNode* chain_node) {
    ordered_chain_nodes_.emplace_back(chain_node);
    int64_t stream_id = chain_node->chain_it()->nodes.front()->GlobalWorkStreamId();
    int64_t chain_id = Global<IDMgr>::Get()->AllocateChainId(stream_id);
    chain_node->set_chain_id(chain_id);
    for (auto& task_node : chain_node->chain_it()->nodes) {
      ordered_task_nodes_.emplace_back(task_node);
    }
  });

  ToDotWithAutoFilePath();
}

ChainNode* ChainGraph::ChainNode4TaskNode(TaskNode* task_node) const {
  auto task2chain_it = task_node2chain_it_.find(task_node);
  CHECK(task2chain_it != task_node2chain_it_.end());
  return task2chain_it->second->chain_node;
}

bool ChainGraph::HasChainEdge(ChainNode* src, ChainNode* dst) const {
  bool has_chain_edge = false;
  for (auto& out_edge : src->out_edges()) {
    if (out_edge->dst_node() == dst) {
      has_chain_edge = true;
      break;
    }
  }
  return has_chain_edge;
}

}  // namespace oneflow
