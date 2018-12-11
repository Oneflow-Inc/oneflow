#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/blocking_counter.h"

namespace oneflow {

namespace {

class ChainMerger final {
 public:
  ChainMerger(const std::vector<TaskNode*>& task_nodes,
              const HashMap<TaskNode*, HashSet<TaskNode*>>& node2ancestors)
      : task_nodes_(task_nodes), node2ancestors_(node2ancestors) {
    InitTaskNode2UId();
    InitChains();
    MergeChains();
  }
  const std::list<Chain>& GetChains() const { return chain_list_; }

 private:
  void InitTaskNode2UId();
  void InitChains();
  void MergeChains();
  bool DoMerge(std::list<ChainIt>& chains, ChainIt rhs);

  bool IsSubset(const ChainIt& lhs, const ChainIt& rhs) const;
  void CarefullySetBitset(std::vector<std::bitset<BITSET_SIZE>>* bitset_vec, int64_t pos);

  int64_t GetTaskUid(TaskNode* task_node) const;
  void UpdateTaskUid(TaskNode* task_node);

  const std::vector<TaskNode*>& task_nodes_;
  std::list<Chain> chain_list_;
  HashMap<TaskNode*, int64_t> task_node2uid_;
  const HashMap<TaskNode*, HashSet<TaskNode*>>& node2ancestors_;
};

int64_t ChainMerger::GetTaskUid(TaskNode* task_node) const {
  auto uid_it = task_node2uid_.find(task_node);
  CHECK(uid_it != task_node2uid_.end());
  return uid_it->second;
}

void ChainMerger::UpdateTaskUid(TaskNode* task_node) {
  auto uid_it = task_node2uid_.find(task_node);
  if (uid_it == task_node2uid_.end()) {
    int64_t new_id = task_node2uid_.size();
    CHECK(task_node2uid_.emplace(task_node, new_id).second);
  }
}

void ChainMerger::InitTaskNode2UId() {
  for (auto& task_node : task_nodes_) {
    UpdateTaskUid(task_node);
    for (auto& ancestor : node2ancestors_.at(task_node)) { UpdateTaskUid(ancestor); }
  }
}

void ChainMerger::InitChains() {
  chain_list_.clear();
  int64_t bitset_num = std::ceil(static_cast<double>(task_node2uid_.size()) / BITSET_SIZE);
  for (const auto& task_node : task_nodes_) {
    chain_list_.emplace_back();
    Chain& cur_chain = chain_list_.back();
    cur_chain.nodes = {task_node};
    cur_chain.stream_area_id =
        std::make_pair(task_node->area_id(), task_node->GlobalWorkStreamId());
    cur_chain.ancestors.resize(bitset_num);
    cur_chain.ancestors_and_this.resize(bitset_num);
    CarefullySetBitset(&(cur_chain.ancestors_and_this), GetTaskUid(task_node));
    for (auto& ancestor : node2ancestors_.at(task_node)) {
      int64_t ancestor_uid = GetTaskUid(ancestor);
      CarefullySetBitset(&(cur_chain.ancestors), ancestor_uid);
      CarefullySetBitset(&(cur_chain.ancestors_and_this), ancestor_uid);
    }
  }
}

bool ChainMerger::DoMerge(std::list<ChainIt>& chains, ChainIt rhs) {
  CHECK_EQ(rhs->nodes.size(), 1);
  // rm kMdUpdtArea chain merge
  if (rhs->nodes.front()->area_id() == kMdUpdtArea) { return false; }
  for (auto chains_it = chains.rbegin(); chains_it != chains.rend(); ++chains_it) {
    ChainIt lhs = *chains_it;
    if (IsSubset(lhs, rhs)) {
      for (TaskNode* node : rhs->nodes) {
        lhs->nodes.push_back(node);
        CarefullySetBitset(&(lhs->ancestors_and_this), GetTaskUid(node));
      }
      return true;
    }
  }
  return false;
}

void ChainMerger::MergeChains() {
  HashMap<std::pair<int64_t, int64_t>, std::list<ChainIt>> stream_area2chains;
  for (auto cur_chain_it = chain_list_.begin(); cur_chain_it != chain_list_.end();) {
    const auto& stream_area_id = cur_chain_it->stream_area_id;
    auto stream_area_it = stream_area2chains.find(stream_area_id);
    if (stream_area_it != stream_area2chains.end()
        && DoMerge(stream_area_it->second, cur_chain_it)) {
      cur_chain_it = chain_list_.erase(cur_chain_it);
    } else {
      stream_area2chains[stream_area_id].push_back(cur_chain_it);
      ++cur_chain_it;
    }
  }
}

void ChainMerger::CarefullySetBitset(std::vector<std::bitset<BITSET_SIZE>>* bitset_vec,
                                     int64_t pos) {
  int64_t index = pos / BITSET_SIZE;
  int64_t remain = pos % BITSET_SIZE;
  bitset_vec->at(index).set(remain);
}

bool ChainMerger::IsSubset(const ChainIt& lhs, const ChainIt& rhs) const {
  CHECK_EQ(lhs->ancestors_and_this.size(), rhs->ancestors_and_this.size());
  int64_t bitset_num = lhs->ancestors_and_this.size();
  for (int64_t i = 0; i < bitset_num; ++i) {
    if (lhs->ancestors_and_this.at(i) != (lhs->ancestors_and_this.at(i) | rhs->ancestors.at(i))) {
      return false;
    }
  }
  return true;
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
  HashMap<int64_t, std::vector<TaskNode*>> machine2tasks;
  HashMap<TaskNode*, HashSet<TaskNode*>> node2ancestors;
  std::vector<std::vector<TaskNode*>> chains;
  GroupTaskNodesByMachineAndCollectAncestors(task_gph, &machine2tasks, &node2ancestors);
  MergeTaskNodes(machine2tasks, node2ancestors, &chains);
  InitChainNode(chains);
  InitChainEdge(chains);
  SetChainId4ChainNode();
  ToDotWithAutoFilePath();
}

void ChainGraph::GroupTaskNodesByMachineAndCollectAncestors(
    const TaskGraph& task_gph, HashMap<int64_t, std::vector<TaskNode*>>* machine2tasks,
    HashMap<TaskNode*, HashSet<TaskNode*>>* node2ancestors) const {
  task_gph.AcyclicTopoForEachNode(TaskGraph::IsBackEdgeByAreaType, [&](TaskNode* node) {
    (*machine2tasks)[node->machine_id()].emplace_back(node);
    CHECK(node2ancestors->emplace(node, HashSet<TaskNode*>()).second);
    // to reduce memory consumption
    if (node->area_id() == kMdUpdtArea) { return; }
    node->ForEachNodeOnInEdge([&](TaskNode* in_node) {
      if (TaskGraph::IsBackEdgeByAreaType(in_node, node)) { return; }
      (*node2ancestors)[node].insert(in_node);
      (*node2ancestors)[node].insert((*node2ancestors)[in_node].begin(),
                                     (*node2ancestors)[in_node].end());
    });
  });
}

void ChainGraph::MergeTaskNodes(const HashMap<int64_t, std::vector<TaskNode*>>& machine2tasks,
                                const HashMap<TaskNode*, HashSet<TaskNode*>>& node2ancestors,
                                std::vector<std::vector<TaskNode*>>* chains) const {
  int64_t machine_num = machine2tasks.size();
  int64_t cpu_num = std::thread::hardware_concurrency();
  int64_t thread_pool_size = std::min(machine_num, cpu_num);
  std::mutex chain_list_mtx;
  BlockingCounter counter(machine_num);
  ThreadPool thread_pool(thread_pool_size);
  for (auto& pair : machine2tasks) {
    thread_pool.AddWork([&]() {
      ChainMerger merger(pair.second, node2ancestors);
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

void ChainGraph::InitChainNode(const std::vector<std::vector<TaskNode*>>& chains) {
  for (auto& chain : chains) {
    ChainNode* chain_node = new ChainNode(chain);
    for (auto& task_node : chain) {
      CHECK(task_node2chain_node_.emplace(task_node, chain_node).second);
    }
    AddAllocatedNode(chain_node);
  }
}

void ChainGraph::InitChainEdge(const std::vector<std::vector<TaskNode*>>& chains) {
  for (auto& chain : chains) {
    for (auto& cur_task_node : chain) {
      auto cur_chain_node = ChainNode4TaskNode(cur_task_node);
      for (auto& task_in_edge : cur_task_node->in_edges()) {
        auto src_task_node = task_in_edge->src_node();
        if (TaskGraph::IsBackEdgeByAreaType(src_task_node, cur_task_node)) { continue; }
        auto src_chain_node = ChainNode4TaskNode(src_task_node);
        if (cur_chain_node == src_chain_node) { continue; }
        if (HasChainEdge(src_chain_node, cur_chain_node)) { continue; }
        Connect(src_chain_node, NewEdge(), cur_chain_node);
      }
    }
  }
}

void ChainGraph::SetChainId4ChainNode() {
  TopoForEachNode([&](ChainNode* chain_node) {
    ordered_chain_nodes_.emplace_back(chain_node);
    int64_t stream_id = chain_node->TaskNodes().front()->GlobalWorkStreamId();
    int64_t chain_id = Global<IDMgr>::Get()->AllocateChainId(stream_id);
    chain_node->SetChainId(chain_id);
  });
}

bool ChainGraph::HasChainEdge(ChainNode* src, ChainNode* dst) const {
  for (auto& out_edge : src->out_edges()) {
    if (out_edge->dst_node() == dst) { return true; }
  }
  return false;
}

}  // namespace oneflow
