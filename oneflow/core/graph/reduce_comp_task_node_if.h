#ifndef ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_

#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "logical_node.h"

namespace oneflow {

class ReduceMemSharingCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceMemSharingCtx);
  ReduceMemSharingCtx(int64_t mem_size, int64_t mem_shared_id)
      : mem_size_(mem_size), mem_shared_id_(mem_shared_id) {}
  ~ReduceMemSharingCtx() = default;

  void EnableMemSharing4Regst(RegstDesc* regst, int64_t offset) const {
    regst->set_enable_mem_sharing(true);
    regst->set_mem_shared_id(static_cast<int32_t>(mem_shared_id_));
    regst->set_mem_shared_offset(offset);
  }

  int64_t Offset4RankingParallelId(const ReduceRankingCtx& ranking, int64_t parallel_id) const {
    if (ranking.TotalSegmentCount() == 1) {
      return 0;
    } else {
      ReduceRankingCtx if_gather = ranking.CtxWithGather();
      return Offset4RankingParallelId(if_gather, parallel_id)
             + ranking.StageRank4ParallelId(parallel_id) * SegmentSize4Ranking(ranking);
    }
  }

  int64_t SegmentSize4Ranking(const ReduceRankingCtx& ranking) const {
    return mem_size_ / ranking.TotalSegmentCount();
  }

 private:
  int64_t mem_size_;
  int64_t mem_shared_id_;
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
  const ReduceRankingCtx& GetRankingCtx() {
    const ReduceLogicalNode* reduce_logical_node =
        dynamic_cast<const ReduceLogicalNode*>(AsCompTaskNode()->logical_node());
    CHECK(reduce_logical_node);
    return reduce_logical_node->ranking_ctx();
  }
  TaskNode* FindPredReduceTaskNodeIf(std::function<bool(TaskNode*)> predicate);
};

int64_t InferRegstSize(const RegstDesc& regst);
void BuildCtrlRegstBetweenReduceCopyNodes(const CompTaskNode* src_reduce,
                                          const CompTaskNode* dst_reduce, int64_t copy_node_num);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_
