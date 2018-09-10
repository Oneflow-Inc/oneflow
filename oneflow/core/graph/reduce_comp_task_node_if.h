#ifndef ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_COMP_TASK_NODE_IF_H_

#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ReduceMemSharingCtx final {
 public:
  ReduceMemSharingCtx(int64_t mem_size, int64_t mem_shared_id)
      : mem_size_(mem_size), mem_shared_id_(mem_shared_id), dims_({1}) {}
  ~ReduceMemSharingCtx() = default;

  void EnableMemSharing4Regst(RegstDesc* regst, int64_t offset) {
    regst->set_enable_mem_sharing(true);
    regst->set_mem_shared_id(static_cast<int32_t>(mem_shared_id_));
    regst->set_mem_shared_offset(offset);
  }

  void Scatter(int64_t size) {
    CHECK_EQ(ReduceSize() % size, 0);
    dims_.push_back(size);
  }

  int64_t Rank4ParallelId(int64_t parallel_id) {
    int64_t rank = parallel_id;
    FOR_RANGE(size_t, i, 0, dims_.size() - 1) { rank /= dims_.at(i); }
    return rank % dims_.back();
  }

  int64_t Offset4ParallelId(int64_t parallel_id) {
    if (ReduceCount() == 1) {
      return 0;
    } else {
      ReduceMemSharingCtx if_gather = CtxIfGatherLast();
      return if_gather.Offset4ParallelId(parallel_id) + Rank4ParallelId(parallel_id) * ReduceSize();
    }
  }

  void Gather(int64_t size) {
    CHECK_EQ(dims_.back(), size);
    CHECK_GT(dims_.size(), 1);
    dims_.pop_back();
  }

  ReduceMemSharingCtx CtxIfGatherLast() {
    ReduceMemSharingCtx ctx = *this;
    ctx.Gather(dims_.back());
    return ctx;
  }

  ReduceMemSharingCtx CtxIfScatter(int64_t size) {
    ReduceMemSharingCtx ctx = *this;
    ctx.Scatter(size);
    return ctx;
  }

  int64_t ReduceCount() {
    int64_t cnt = 1;
    for (const int64_t dim : dims_) { cnt *= dim; }
    return cnt;
  }

  int64_t LastCount() { return dims_.back(); }

  int64_t ReduceSize() { return mem_size_ / ReduceCount(); }

 private:
  int64_t mem_size_;
  int64_t mem_shared_id_;
  std::vector<int64_t> dims_;
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
