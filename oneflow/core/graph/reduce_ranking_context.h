#ifndef ONEFLOW_CORE_GRAPH_REDUCE_RANKING_CONTEXT_H_
#define ONEFLOW_CORE_GRAPH_REDUCE_RANKING_CONTEXT_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class ReduceRankingCtx final {
 public:
  ReduceRankingCtx() : segment_count_of_each_stage_({1}) {}
  ~ReduceRankingCtx() = default;

  int64_t StageRank4ParallelId(int64_t parallel_id) const {
    int64_t rank = parallel_id;
    FOR_RANGE(size_t, i, 0, segment_count_of_each_stage_.size() - 1) {
      rank /= segment_count_of_each_stage_.at(i);
    }
    return rank % segment_count_of_each_stage_.back();
  }

  ReduceRankingCtx CtxWithGather() const {
    CHECK_GT(segment_count_of_each_stage_.size(), 1);
    std::vector<int64_t> scatters = segment_count_of_each_stage_;
    scatters.pop_back();
    return ReduceRankingCtx(scatters);
  }

  ReduceRankingCtx CtxWithScatter(int64_t size) const {
    std::vector<int64_t> scatters = segment_count_of_each_stage_;
    scatters.push_back(size);
    return ReduceRankingCtx(scatters);
  }

  int64_t TotalSegmentCount() const {
    int64_t cnt = 1;
    for (const int64_t dim : segment_count_of_each_stage_) { cnt *= dim; }
    return cnt;
  }

  int64_t StageSegmentCount() const { return segment_count_of_each_stage_.back(); }

 private:
  explicit ReduceRankingCtx(std::vector<int64_t> scatters)
      : segment_count_of_each_stage_(std::move(scatters)) {}
  std::vector<int64_t> segment_count_of_each_stage_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REDUCE_RANKING_CONTEXT_H_
