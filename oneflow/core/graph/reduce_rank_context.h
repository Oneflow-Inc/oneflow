#ifndef ONEFLOW_CORE_GRAPH_REDUCE_RANK_CONTEXT_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_RANK_CONTEXT_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class ReduceRankCtx final {
 public:
  ReduceRankCtx() : segment_count_of_each_stage_({1}) {}
  ~ReduceRankCtx() = default;

  int64_t StageRank4ParallelId(int64_t parallel_id) const {
    int64_t rank = parallel_id;
    FOR_RANGE(size_t, i, 0, segment_count_of_each_stage_.size() - 1) {
      rank /= segment_count_of_each_stage_.at(i);
    }
    return rank % segment_count_of_each_stage_.back();
  }

  int64_t RankSet4ParallelId(int64_t parallel_id) const {
    if (Depth() == 1) {
      return parallel_id;
    } else {
      ReduceRankCtx if_gather = CtxWithGather();
      return (parallel_id / TotalSegmentCount() * if_gather.TotalSegmentCount()
              + parallel_id % if_gather.TotalSegmentCount());
    }
  }

  ReduceRankCtx CtxWithGather() const {
    CHECK_GT(segment_count_of_each_stage_.size(), 1);
    std::vector<int64_t> segment_counts = segment_count_of_each_stage_;
    segment_counts.pop_back();
    return ReduceRankCtx(segment_counts);
  }

  ReduceRankCtx CtxWithScatter(int64_t size) const {
    std::vector<int64_t> segment_counts = segment_count_of_each_stage_;
    segment_counts.push_back(size);
    return ReduceRankCtx(segment_counts);
  }

  int64_t TotalSegmentCount() const {
    int64_t cnt = 1;
    for (const int64_t dim : segment_count_of_each_stage_) { cnt *= dim; }
    return cnt;
  }

  int64_t StageSegmentCount() const { return segment_count_of_each_stage_.back(); }

  int64_t Depth() const { return segment_count_of_each_stage_.size(); }

 private:
  explicit ReduceRankCtx(std::vector<int64_t> counts)
      : segment_count_of_each_stage_(std::move(counts)) {}
  std::vector<int64_t> segment_count_of_each_stage_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_RANK_CONTEXT_H_
