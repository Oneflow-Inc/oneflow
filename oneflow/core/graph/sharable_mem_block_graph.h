#ifndef ONEFLOW_CORE_GRAPH_SHARABLE_MEM_BLOCK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_SHARABLE_MEM_BLOCK_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/register/register_desc.pb.h"
#include "oneflow/core/graph/plan_task_graph.h"

namespace oneflow {

class SharableMemBlockEdge;

class SharableMemBlockNode final : public Node<SharableMemBlockNode, SharableMemBlockEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SharableMemBlockNode);
  SharableMemBlockNode(int64_t chain_id, const MemBlock& mem_block,
                       const HashSet<const RegstDescProto*>& regst_descs,
                       const PlanTaskGraph& plan_task_graph)
      : chain_id_(chain_id),
        mem_block_(mem_block),
        regst_descs_(regst_descs.begin(), regst_descs.end()) {}

  ~SharableMemBlockNode() = default;

  int64_t chain_id() const { return chain_id_; }
  const std::vector<const RegstDescProto*>& regst_descs() const { return regst_descs_; }
  const MemBlock& mem_block() const { return mem_block_; }

 private:
  const int64_t chain_id_;
  const MemBlock mem_block_;
  const std::vector<const RegstDescProto*> regst_descs_;
};

class SharableMemBlockEdge final : public Edge<SharableMemBlockNode, SharableMemBlockEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SharableMemBlockEdge);
  SharableMemBlockEdge() = default;
  ~SharableMemBlockEdge() = default;
};

class SharableMemBlockGraph final : public Graph<const SharableMemBlockNode, SharableMemBlockEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SharableMemBlockGraph);
  SharableMemBlockGraph(const PlanTaskGraph& plan_task_gph,
                        const std::function<bool(const RegstDescProto&)>& IsSharable);
  ~SharableMemBlockGraph() = default;

  void ForEachSourceNodeGroup(
      const std::function<int64_t(const SharableMemBlockNode*)>& GroupBy,
      const std::function<void(const std::vector<const SharableMemBlockNode*>&)>& Handler) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_SHARABLE_MEM_BLOCK_GRAPH_H_
