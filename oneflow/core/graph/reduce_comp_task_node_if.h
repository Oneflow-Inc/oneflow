#ifndef ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_

#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ReduceMemSharingCtx final {
 public:
  ReduceMemSharingCtx(int64_t mem_size, int64_t mem_shared_id)
      : mem_size_(mem_size), mem_shared_id_(mem_shared_id), scatter_segment_counts_({1}) {}
  ~ReduceMemSharingCtx() = default;

  void EnableMemSharing4Regst(RegstDesc* regst, int64_t offset) const {
    regst->set_enable_mem_sharing(true);
    regst->set_mem_shared_id(static_cast<int32_t>(mem_shared_id_));
    regst->set_mem_shared_offset(offset);
  }

  void DoScatter(int64_t size) {
    CHECK_EQ(StageSegmentSize() % size, 0);
    scatter_segment_counts_.push_back(size);
  }

  int64_t StageRank4ParallelId(int64_t parallel_id) const {
    int64_t rank = parallel_id;
    FOR_RANGE(size_t, i, 0, scatter_segment_counts_.size() - 1) {
      rank /= scatter_segment_counts_.at(i);
    }
    return rank % scatter_segment_counts_.back();
  }

  int64_t Offset4ParallelId(int64_t parallel_id) const {
    if (TotalSegmentCount() == 1) {
      return 0;
    } else {
      ReduceMemSharingCtx if_gather = CtxWithGather();
      return if_gather.Offset4ParallelId(parallel_id)
             + StageRank4ParallelId(parallel_id) * StageSegmentSize();
    }
  }

  void DoGather(int64_t size) {
    CHECK_EQ(scatter_segment_counts_.back(), size);
    CHECK_GT(scatter_segment_counts_.size(), 1);
    scatter_segment_counts_.pop_back();
  }

  ReduceMemSharingCtx CtxWithGather() const {
    ReduceMemSharingCtx ctx = *this;
    ctx.DoGather(scatter_segment_counts_.back());
    return ctx;
  }

  ReduceMemSharingCtx CtxWithScatter(int64_t size) const {
    ReduceMemSharingCtx ctx = *this;
    ctx.DoScatter(size);
    return ctx;
  }

  int64_t TotalSegmentCount() const {
    int64_t cnt = 1;
    for (const int64_t dim : scatter_segment_counts_) { cnt *= dim; }
    return cnt;
  }

  int64_t StageSegmentCount() const { return scatter_segment_counts_.back(); }

  int64_t StageSegmentSize() const { return mem_size_ / TotalSegmentCount(); }

 private:
  int64_t mem_size_;
  int64_t mem_shared_id_;
  std::vector<int64_t> scatter_segment_counts_;
};

class ReduceCompTaskNodeIf {
 public:
  virtual ~ReduceCompTaskNodeIf() = default;
  virtual void EnableMemSharingInReduce(ReduceMemSharingCtx*) = 0;

 protected:
  struct EdgeInfo {
    TaskEdge* edge;
    int64_t order;
  };
  void SortEdges(std::vector<EdgeInfo>* edge_infos) {
    std::sort((*edge_infos).begin(), (*edge_infos).end(),
              [](const EdgeInfo& lhs, const EdgeInfo& rhs) { return lhs.order < rhs.order; });
  }
};

int64_t InferRegstSize(const RegstDesc& regst);
void BuildCtrlRegstBetweenReduceCopyNodes(const CompTaskNode* src_reduce,
                                          const CompTaskNode* dst_reduce, int64_t copy_node_num);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_
