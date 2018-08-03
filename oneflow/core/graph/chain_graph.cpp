#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/blocking_counter.h"

namespace oneflow {

namespace {

class ChainMerger final {
 public:
  ChainMerger(const std::vector<TaskNode*>& task_nodes) : task_nodes_(task_nodes) {
    InitTaskNode2UId();
    InitChains();
    MergeChains();
  }
  const std::list<Chain>& GetChains() const { return chain_list_; }

 private:
  void InitTaskNode2UId();
  void InitChains();
  bool DoMerge(std::list<ChainIt>& chains, ChainIt rhs);
  bool TryMerge();
  void MergeChains();
  int64_t get_task_uid(TaskNode* task_node) const {
    auto uid_it = task_node2uid_.find(task_node);
    CHECK(uid_it != task_node2uid_.end());
    return uid_it->second;
  }
  void update_task_uid(TaskNode* task_node) {
    auto uid_it = task_node2uid_.find(task_node);
    if (uid_it == task_node2uid_.end()) {
      int64_t new_id = task_node2uid_.size();
      CHECK(task_node2uid_.emplace(task_node, new_id).second);
    }
  }

  const std::vector<TaskNode*>& task_nodes_;
  std::list<Chain> chain_list_;
  HashMap<TaskNode*, int64_t> task_node2uid_;
};

void ChainMerger::InitTaskNode2UId() {
  for (auto& task_node : task_nodes_) {
    update_task_uid(task_node);
    for (auto& ancestor : task_node->ancestors()) { update_task_uid(ancestor); }
  }
  CHECK_LT(task_node2uid_.size(), MAX_ANCESTOR_NUM);
}

void ChainMerger::InitChains() {
  chain_list_.clear();
  for (const auto& task_node : task_nodes_) {
    chain_list_.emplace_back();
    Chain& cur_chain = chain_list_.back();
    cur_chain.nodes = {task_node};
    cur_chain.area_id = task_node->area_id();
    cur_chain.stream_id = task_node->GlobalWorkStreamId();
    for (auto& node : cur_chain.nodes) { cur_chain.ancestors_and_this.set(get_task_uid(node)); }
    for (auto& node : task_node->ancestors()) {
      cur_chain.ancestors.set(get_task_uid(node));
      cur_chain.ancestors_and_this.set(get_task_uid(node));
    }
  }
}

bool ChainMerger::DoMerge(std::list<ChainIt>& chains, ChainIt rhs) {
  for (auto chains_it = chains.rbegin(); chains_it != chains.rend(); ++chains_it) {
    ChainIt lhs = *chains_it;
    if (lhs->ancestors_and_this == (lhs->ancestors_and_this | rhs->ancestors)) {
      for (TaskNode* node : rhs->nodes) {
        lhs->nodes.push_back(node);
        lhs->ancestors_and_this.set(get_task_uid(node));
      }
      return true;
    }
  }
  return false;
}

bool ChainMerger::TryMerge() {
  HashMap<std::pair<int64_t, int64_t>, std::list<ChainIt>> stream_area2chains;
  bool merge_happened = false;
  for (auto cur_chain_it = chain_list_.begin(); cur_chain_it != chain_list_.end();) {
    std::pair<int64_t, int64_t> stream_area_id = {cur_chain_it->stream_id, cur_chain_it->area_id};
    auto stream_area_it = stream_area2chains.find(stream_area_id);
    if (stream_area_it != stream_area2chains.end()
        && DoMerge(stream_area_it->second, cur_chain_it)) {
      cur_chain_it = chain_list_.erase(cur_chain_it);
      merge_happened = true;
    } else {
      stream_area2chains[stream_area_id].push_back(cur_chain_it);
      ++cur_chain_it;
    }
  }
  return merge_happened;
}

void ChainMerger::MergeChains() {
  while (TryMerge()) {}
}

}  // namespace

std::string ChainNode::VisualStr() const {
  std::stringstream ss;
  ss << "chain_id:" << chain_id_ << "\\n";
  for (auto& task_node : task_nodes_) {
    ss << TaskType_Name(task_node->GetTaskType()) << ":" << task_node->task_id() << "\\n";
  }
  return ss.str();
}

ChainGraph::ChainGraph(const TaskGraph& task_gph) : task_gph_(task_gph) {
  std::vector<TaskNode*> ordered_task_nodes;
  HashMap<int64_t, std::vector<TaskNode*>> machine2tasks;
  std::vector<std::vector<TaskNode*>> chains;

  task_gph.AcyclicTopoForEachNode([&](TaskNode* node) { ordered_task_nodes.emplace_back(node); });
  GroupTaskNodesByMachine(ordered_task_nodes, &machine2tasks);
  MergeTaskNodes(machine2tasks, &chains);

  for (auto& task_nodes_in_chain : chains) {
    ChainNode* chain_node = new ChainNode(task_nodes_in_chain);
    for (auto& task_node : task_nodes_in_chain) {
      CHECK(task_node2chain_node_.emplace(task_node, chain_node).second);
    }
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
    int64_t stream_id = chain_node->task_nodes().front()->GlobalWorkStreamId();
    int64_t chain_id = Global<IDMgr>::Get()->AllocateChainId(stream_id);
    chain_node->set_chain_id(chain_id);
    for (auto& task_node : chain_node->task_nodes()) {
      ordered_task_nodes_.emplace_back(task_node);
    }
  });

  ToDotWithAutoFilePath();
}

void ChainGraph::GroupTaskNodesByMachine(const std::vector<TaskNode*>& ordered_task_nodes,
                                         HashMap<int64_t, std::vector<TaskNode*>>* machine2tasks) {
  for (auto& task_node : ordered_task_nodes) {
    int64_t machine_id = task_node->machine_id();
    auto machine_it = machine2tasks->find(machine_id);
    if (machine_it != machine2tasks->end()) {
      machine_it->second.push_back(task_node);
    } else {
      std::vector<TaskNode*> task_nodes{task_node};
      CHECK(machine2tasks->emplace(machine_id, task_nodes).second);
    }
  }
}
void ChainGraph::MergeTaskNodes(const HashMap<int64_t, std::vector<TaskNode*>>& machine2tasks,
                                std::vector<std::vector<TaskNode*>>* chains) {
  int64_t machine_num = machine2tasks.size();
  int64_t cpu_num = std::thread::hardware_concurrency();
  int64_t thread_pool_size = std::min(machine_num, cpu_num);
  std::mutex chain_list_mtx;
  BlockingCounter counter(machine_num);
  ThreadPool thread_pool(thread_pool_size);
  for (auto& pair : machine2tasks) {
    thread_pool.AddWork([&]() {
      ChainMerger merger(pair.second);
      auto& cur_chain_list = merger.GetChains();
      {
        std::unique_lock<std::mutex> guard(chain_list_mtx);
        for (const auto& chain : cur_chain_list) { chains->emplace_back(chain.nodes); }
      }
      counter.Decrease();
    });
  }
  counter.WaitUntilCntEqualZero();
}

ChainNode* ChainGraph::ChainNode4TaskNode(TaskNode* task_node) const {
  auto task2chain_it = task_node2chain_node_.find(task_node);
  CHECK(task2chain_it != task_node2chain_node_.end());
  return task2chain_it->second;
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
