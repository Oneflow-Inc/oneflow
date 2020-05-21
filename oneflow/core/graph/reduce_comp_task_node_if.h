#ifndef ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_

#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class ReduceMemSharingCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceMemSharingCtx);
  ReduceMemSharingCtx(int64_t mem_size, int64_t mem_block_id)
      : mem_size_(mem_size), mem_block_id_(mem_block_id) {}
  ~ReduceMemSharingCtx() = default;

  void EnableMemSharing4Regst(RegstDesc* regst, int64_t offset) const {
    regst->set_enable_reuse_mem(false);
    regst->set_mem_block_id(static_cast<int32_t>(mem_block_id_));
    regst->set_mem_block_offset(offset);
  }

  int64_t Offset4RankCtxParallelId(const ReduceRankCtx& rank_ctx, int64_t parallel_id) const {
    if (rank_ctx.TotalSegmentCount() == 1) {
      return 0;
    } else {
      ReduceRankCtx if_gather = rank_ctx.CtxWithGather();
      return Offset4RankCtxParallelId(if_gather, parallel_id)
             + rank_ctx.Rank4ParallelId(parallel_id) * SegmentSize4RankCtx(rank_ctx);
    }
  }

  int64_t SegmentSize4RankCtx(const ReduceRankCtx& rank_ctx) const {
    CHECK_EQ(mem_size_ % rank_ctx.TotalSegmentCount(), 0);
    return mem_size_ / rank_ctx.TotalSegmentCount();
  }

 private:
  int64_t mem_size_;
  int64_t mem_block_id_;
};

class ReduceCompTaskNodeIf {
 public:
  virtual ~ReduceCompTaskNodeIf() = default;
  virtual void EnableMemSharingInReduce(const ReduceMemSharingCtx& ctx) = 0;

 protected:
  struct EdgeInfo {
    TaskEdge* edge;
    int64_t order;
  };
  void SortEdges(std::vector<EdgeInfo>* edge_infos) {
    std::sort((*edge_infos).begin(), (*edge_infos).end(),
              [](const EdgeInfo& lhs, const EdgeInfo& rhs) { return lhs.order < rhs.order; });
  }
  CompTaskNode* AsCompTaskNode() { return dynamic_cast<CompTaskNode*>(this); }
  const ReduceRankCtx& GetRankCtx() {
    const ReduceLogicalNode* reduce_logical_node =
        dynamic_cast<const ReduceLogicalNode*>(AsCompTaskNode()->logical_node());
    CHECK(reduce_logical_node);
    return reduce_logical_node->rank_ctx();
  }
  TaskNode* FindPredReduceTaskNodeIf(std::function<bool(TaskNode*)> predicate);
};

int64_t InferRegstSize(const RegstDesc& regst);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_
