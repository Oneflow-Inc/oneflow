#include "oneflow/core/graph/sharable_mem_block_graph.h"
#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

namespace {

bool IsConsumersAndProducerInSameChain(const RegstDescProto& regst_desc,
                                       const PlanTaskGraph& plan_task_graph) {
  auto ChainId4TaskId = [&](int64_t task_id) {
    return plan_task_graph.TaskProto4TaskId(task_id)->task_set_info().chain_id();
  };
  int64_t producer_chain_id = ChainId4TaskId(regst_desc.producer_task_id());
  for (int64_t consumer_task_id : regst_desc.consumer_task_id()) {
    if (ChainId4TaskId(consumer_task_id) != producer_chain_id) { return false; }
  }
  return true;
}

}  // namespace

SharableMemBlockGraph::SharableMemBlockGraph(
    const PlanTaskGraph& plan_task_gph,
    const std::function<bool(const RegstDescProto&)>& IsSharable) {
  auto ForEachSharableChainRegstDesc =
      [&](const std::function<void(int64_t, const RegstDescProto&)>& Handler) {
        HashSet<int64_t> mem_block_ids_check;
        for (const TaskProto& task : plan_task_gph.plan().task()) {
          for (const auto& pair : task.produced_regst_desc()) {
            if (IsConsumersAndProducerInSameChain(pair.second, plan_task_gph)
                && IsSharable(pair.second)) {
              int32_t idx = 0;
              for (const auto& mem_block : pair.second.mem_block_hierarchy()) {
                if (idx++ == 0) { CHECK(mem_block_ids_check.emplace(mem_block.block_id()).second); }
                Handler(task.task_set_info().chain_id(), pair.second);
              }
            }
          }
        }
      };
  HashMap<std::pair<int64_t, MemBlock>, std::vector<const RegstDescProto*>>
      chain_id7mem_block2regst_descs;
  ForEachSharableChainRegstDesc([&](int64_t chain_id, const RegstDescProto& regst_desc) {
    for (const auto& mem_block : regst_desc.mem_block_hierarchy()) {
      chain_id7mem_block2regst_descs[std::make_pair(chain_id, mem_block)].push_back(&regst_desc);
    }
  });
  HashMap<std::pair<int64_t, MemBlock>, SharableMemBlockNode*> chain_id7mem_block2node;
  for (const auto& pair : chain_id7mem_block2regst_descs) {
    auto* node =
        new SharableMemBlockNode(pair.first.first, pair.first.second, pair.second, plan_task_gph);
    AddAllocatedNode(node);
    CHECK(chain_id7mem_block2node.emplace(pair.first, node).second);
  }
  HashSet<const SharableMemBlockNode*> connected_children;
  ForEachSharableChainRegstDesc([&](int64_t chain_id, const RegstDescProto& regst_desc) {
    SharableMemBlockNode* child = nullptr;
    for (const auto& mem_block : regst_desc.mem_block_hierarchy()) {
      auto* parent = chain_id7mem_block2node.at(std::make_pair(chain_id, mem_block));
      if (child != nullptr && connected_children.find(child) == connected_children.end()) {
        Connect(parent, NewEdge(), child);
        CHECK(connected_children.emplace(child).second);
      }
      child = parent;
    }
  });
}

void SharableMemBlockGraph::ForEachSourceNodeGroup(
    const std::function<int64_t(const SharableMemBlockNode*)>& GroupBy,
    const std::function<void(const std::vector<const SharableMemBlockNode*>&)>& Handler) const {
  HashMap<int64_t, std::vector<const SharableMemBlockNode*>> chain_id2source_nodes;
  for (const SharableMemBlockNode* source : source_nodes()) {
    chain_id2source_nodes[GroupBy(source)].push_back(source);
  }
  for (const auto& pair : chain_id2source_nodes) { Handler(pair.second); }
}

}  // namespace oneflow
