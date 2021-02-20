/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/job/global_for.h"

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
        std::make_pair(task_node->AreaId4ChainMerge(), task_node->GlobalWorkStreamId());
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
  if (rhs->nodes.front()->AreaId4ChainMerge() == kMdUpdtArea) { return false; }
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

bool IsForwardOnlyTaskNode(TaskNode* node) {
  auto* fw_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
  if (fw_node == nullptr) { return true; }
  return fw_node->HasBackwardCompTaskNode() == false;
};

bool NoOutRegstConsumedByBwNode(TaskNode* node) {
  auto* fw_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
  if (fw_node == nullptr) { return false; }
  for (TaskEdge* edge : fw_node->out_edges()) {
    auto* fw_consumer = dynamic_cast<NormalForwardCompTaskNode*>(edge->dst_node());
    if (fw_consumer == nullptr) { return false; }
    if (fw_consumer->HasBackwardCompTaskNode()) { return false; }
  }
  return true;
};

void CollectIgnoreTaskEdgesInFirstMergedChains(const std::vector<std::vector<TaskNode*>>& chains,
                                               HashSet<TaskEdge*>* ignore_edges) {
  auto HasGpuVariableOpInChain = [&](const std::vector<TaskNode*>& chain) -> bool {
    for (TaskNode* node : chain) {
      auto* fw_node = dynamic_cast<NormalForwardCompTaskNode*>(node);
      if (fw_node == nullptr) { continue; }
      if (fw_node->logical_node()->op_vec().size() != 1) { continue; }
      const auto& src_op = *fw_node->logical_node()->SoleOp();
      DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(src_op.op_conf().device_tag()));
      if (src_op.op_conf().has_variable_conf() && device_type == DeviceType::kGPU) { return true; }
    }
    return false;
  };
  for (auto& chain : chains) {
    if (HasGpuVariableOpInChain(chain)) {
      HashSet<TaskNode*> nodes_in_chain(chain.begin(), chain.end());
      for (TaskNode* node : chain) {
        for (TaskEdge* out_edge : node->out_edges()) {
          if (nodes_in_chain.find(out_edge->dst_node()) == nodes_in_chain.end()) {
            ignore_edges->insert(out_edge);
          }
        }
      }
    }
  }
}

}  // namespace

std::string ChainNode::VisualStr() const {
  std::stringstream ss;
  ss << "chain_id:" << chain_id_ << "\\n";
  for (const auto* task_node : task_nodes_) { ss << task_node->VisualStr(); }
  return ss.str();
}

ChainGraph::ChainGraph(const TaskGraph& task_gph) : task_gph_(task_gph) {
  HashMap<int64_t, std::vector<TaskNode*>> machine2tasks;
  HashMap<TaskNode*, HashSet<TaskNode*>> node2ancestors;
  std::vector<std::vector<TaskNode*>> chains;
  GroupTaskNodesByMachine(task_gph, &machine2tasks);
  // do first merge
  CollectTaskNodeAncestors(task_gph, &node2ancestors, nullptr);
  MergeTaskNodes(machine2tasks, node2ancestors, &chains);
  // collect ignore task edges (variable chain out edges)
  HashSet<TaskEdge*> ignore_edges;
  CollectIgnoreTaskEdgesInFirstMergedChains(chains, &ignore_edges);
  if (!ignore_edges.empty()) {
    // do second merge
    node2ancestors.clear();
    chains.clear();
    CollectTaskNodeAncestors(task_gph, &node2ancestors, &ignore_edges);
    MergeTaskNodes(machine2tasks, node2ancestors, &chains);
  }

  for (auto& task_nodes : chains) { PrioritizeUntrainableTaskNode(&task_nodes); }
  InitChainNode(chains);
  InitChainEdge(chains);
  // NOTE(chengcheng): Remove this check because:
  //   Even if there is a cycle in chain graph, there is no problem.
  // CheckNoCycle();
  SetChainId4ChainNode();
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) {
    ToDotWithFilePath(JoinPath("dot", TypeName(), GlobalJobDesc().job_name() + ".dot"));
  }
}

void ChainGraph::CheckNoCycle() const {
  auto scc = FindFirstNontrivialSCC();
  if (scc) {
    std::string job_id = std::to_string(GlobalJobDesc().job_id());
    auto* ptr = scc.get();
    auto OnCycle = [ptr](ChainNode* chain_node) { return ptr->find(chain_node) != ptr->end(); };
    const auto& chain_graph_filename = "job" + job_id + "_cycle_chain_graph.dot";
    ToDotWithFilePath(OnCycle, [](ChainEdge*) { return true; }, chain_graph_filename);

    HashMap<int64_t, int32_t> task_id2color = {};
    int32_t chain_node_cnt = 0;
    for (ChainNode* chain_node : *ptr) {
      chain_node_cnt++;
      for (TaskNode* task_node : chain_node->TaskNodes()) {
        task_id2color.emplace(task_node->task_id(), chain_node_cnt);
      }
    }
    const std::function<std::string(TaskNode*)> ColorNode = [&](TaskNode* t) {
      auto color_it = task_id2color.find(t->task_id());
      if (color_it != task_id2color.end()) {
        return ", style=filled, colorscheme=set312, fillcolor=" + std::to_string(color_it->second);
      } else {
        return std::string("");
      }
    };
    const std::function<std::string(TaskEdge*)> ColorEdge = [&](TaskEdge* te) {
      auto src_color_it = task_id2color.find(te->src_node()->task_id());
      auto dst_color_it = task_id2color.find(te->dst_node()->task_id());
      if (src_color_it != task_id2color.end() && dst_color_it != task_id2color.end()) {
        return ", style=filled, colorscheme=set312, color=" + std::to_string(task_id2color.size());
      } else {
        return std::string("");
      }
    };
    const std::string colored_task_graph_filename =
        "optimized_dlnet_" + job_id + "_highlighted_cycle_task_nodes_in_chain_graph.dot";
    task_gph_.ToDotWithFilePath(ColorNode, ColorEdge, colored_task_graph_filename);

    HashSet<const TaskNode*> tasks;
    for (const auto* chain_node : *scc) {
      for (const TaskNode* task_node : chain_node->TaskNodes()) {
        CHECK(tasks.emplace(task_node).second);
      }
    }
    auto TaskOnCycle = [&](TaskNode* task) { return tasks.find(task) != tasks.end(); };
    const auto& task_gph_filename = "job" + job_id + "_cycle_task_graph.dot";
    task_gph_.ToDotWithFilePath(TaskOnCycle, [](TaskEdge*) { return true; }, task_gph_filename);
    LOG(FATAL) << "cycle in graph detected, please check:\n"
               << colored_task_graph_filename << "\n"
               << task_gph_filename << "\n"
               << chain_graph_filename;
  }
}

void ChainGraph::GroupTaskNodesByMachine(
    const TaskGraph& task_gph, HashMap<int64_t, std::vector<TaskNode*>>* machine2tasks) const {
  task_gph.AcyclicTopoForEachNode(
      [&](TaskNode* node) { (*machine2tasks)[node->machine_id()].emplace_back(node); });
}

void ChainGraph::CollectTaskNodeAncestors(const TaskGraph& task_gph,
                                          HashMap<TaskNode*, HashSet<TaskNode*>>* node2ancestors,
                                          HashSet<TaskEdge*>* ignore_edges) const {
  task_gph.AcyclicTopoForEachNode([&](TaskNode* node) {
    CHECK(node2ancestors->emplace(node, HashSet<TaskNode*>()).second);
    // to reduce memory consumption
    if (node->GetTaskType() == TaskType::kTick) { return; }
    for (TaskEdge* in_edge : node->in_edges()) {
      if (ignore_edges && ignore_edges->find(in_edge) != ignore_edges->end()) { continue; }
      TaskNode* in_node = in_edge->src_node();
      if (in_node->GetTaskType() == TaskType::kTick) { continue; }
      (*node2ancestors)[node].insert(in_node);
      (*node2ancestors)[node].insert((*node2ancestors)[in_node].begin(),
                                     (*node2ancestors)[in_node].end());
    }
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

void ChainGraph::PrioritizeUntrainableTaskNode(std::vector<TaskNode*>* task_nodes) const {
  HashSet<TaskNode*> task_nodes_set(task_nodes->begin(), task_nodes->end());
  auto IsInSubset = [&](TaskNode* node) {
    return task_nodes_set.find(node) != task_nodes_set.end();
  };
  auto ForEachInNode = [&](TaskNode* node, const std::function<void(TaskNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](TaskNode* node_on_in_edge) {
      if (IsInSubset(node_on_in_edge)) { Handler(node_on_in_edge); }
    });
  };
  auto ForEachOutNode = [&](TaskNode* node, const std::function<void(TaskNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](TaskNode* node_on_out_edge) {
      if (IsInSubset(node_on_out_edge)) { Handler(node_on_out_edge); }
    });
  };
  auto IsSourceNode = [&](TaskNode* node) {
    int32_t in_node_num = 0;
    ForEachInNode(node, [&](TaskNode* in_node) { ++in_node_num; });
    return in_node_num == 0;
  };
  std::list<TaskNode*> starts;
  for (TaskNode* node : task_nodes_set) {
    if (IsSourceNode(node)) { starts.push_back(node); }
  }
  task_nodes->clear();
  auto IsPrior = [&](TaskNode* node) {
    return IsForwardOnlyTaskNode(node) && NoOutRegstConsumedByBwNode(node);
  };
  PartialPriorTopoForEachNode(starts, ForEachInNode, ForEachOutNode, IsPrior,
                              [&](TaskNode* node) { task_nodes->push_back(node); });
  HashSet<TaskNode*> task_nodes_set_check(task_nodes->begin(), task_nodes->end());
  CHECK(task_nodes_set == task_nodes_set_check);
}

void ChainGraph::PartialPriorTopoForEachNode(
    const std::list<TaskNode*> starts,
    const std::function<void(TaskNode*, const std::function<void(TaskNode*)>&)>& ForEachInNode,
    const std::function<void(TaskNode*, const std::function<void(TaskNode*)>&)>& ForEachOutNode,
    const std::function<bool(TaskNode*)>& IsPrior,
    const std::function<void(TaskNode*)>& Handler) const {
  // collect prior nodes
  HashSet<TaskNode*> prior_nodes;
  auto IsTaskNodePrior = [&](TaskNode* node) {
    if (!IsPrior(node)) { return false; }
    bool is_prior = true;
    ForEachInNode(node, [&](TaskNode* in_node) {
      is_prior = is_prior && (prior_nodes.find(in_node) != prior_nodes.end());
    });
    return is_prior;
  };
  std::list<TaskNode*> nodes;
  task_gph_.TopoForEachNode(starts, ForEachInNode, ForEachOutNode, [&](TaskNode* node) {
    if (IsTaskNodePrior(node)) { CHECK(prior_nodes.emplace(node).second); }
    nodes.push_back(node);
  });
  // travel prior nodes;
  auto ForEachPriorInNode = [&](TaskNode* node, const std::function<void(TaskNode*)>& Handler) {
    ForEachInNode(node, [&](TaskNode* in_node) {
      if (prior_nodes.find(in_node) != prior_nodes.end()) { Handler(in_node); }
    });
  };
  auto ForEachPriorOutNode = [&](TaskNode* node, const std::function<void(TaskNode*)>& Handler) {
    ForEachOutNode(node, [&](TaskNode* out_node) {
      if (prior_nodes.find(out_node) != prior_nodes.end()) { Handler(out_node); }
    });
  };
  std::list<TaskNode*> prior_starts;
  for (TaskNode* start : starts) {
    if (IsTaskNodePrior(start)) { prior_starts.push_back(start); }
  }
  task_gph_.DfsTopoForEachNodeSortByDistanceToSink(prior_starts, ForEachPriorInNode,
                                                   ForEachPriorOutNode, Handler);
  // travel other nodes ;
  task_gph_.DfsTopoForEachNodeSortByDistanceToSink(
      starts, ForEachInNode, ForEachOutNode, [&](TaskNode* node) {
        if (prior_nodes.find(node) == prior_nodes.end()) { Handler(node); }
      });
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
        auto src_chain_node = ChainNode4TaskNode(src_task_node);
        if (cur_chain_node == src_chain_node) { continue; }
        if (HasChainEdge(src_chain_node, cur_chain_node)) { continue; }
        Connect(src_chain_node, NewEdge(), cur_chain_node);
      }
    }
  }
}

void ChainGraph::SetChainId4ChainNode() {
  ForEachNode([&](ChainNode* chain_node) {
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
