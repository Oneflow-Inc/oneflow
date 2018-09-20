#include "oneflow/core/graph/reduce_rank_context.h"

namespace oneflow {

TEST(ReduceRankCtx, ScatterGatter) {
  ReduceRankCtx with_gather =
      ReduceRankCtx().CtxWithScatter(4).CtxWithScatter(3).CtxWithScatter(2).CtxWithGather();
  ReduceRankCtx with_out_gather = ReduceRankCtx().CtxWithScatter(4).CtxWithScatter(3);
  ASSERT_EQ(with_gather, with_out_gather);
}

TEST(ReduceRankCtx, test) {
  ReduceRankCtx ctx;
  ASSERT_EQ(ctx.Depth(), 1);
  ASSERT_EQ(ctx.TotalSegmentCount(), 1);
  ASSERT_EQ(ctx.StageSegmentCount(), 1);

  for (int64_t i = 0; i < 24; i++) {
    ASSERT_EQ(ctx.StageRank4ParallelId(i), 0);
    ASSERT_EQ(ctx.RankSet4ParallelId(i), i);
  }

  ReduceRankCtx ctx_24 = ReduceRankCtx().CtxWithScatter(24);
  ASSERT_EQ(ctx_24.Depth(), 2);
  ASSERT_EQ(ctx_24.TotalSegmentCount(), 24);
  ASSERT_EQ(ctx_24.StageSegmentCount(), 24);
  for (int64_t i = 0; i < 24; i++) {
    ASSERT_EQ(ctx_24.StageRank4ParallelId(i), i);
    ASSERT_EQ(ctx_24.RankSet4ParallelId(i), 0);
  }

  ReduceRankCtx ctx_4 = ReduceRankCtx().CtxWithScatter(4);
  std::vector<int64_t> ctx_4_ranks = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                                      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  std::vector<int64_t> ctx_4_sets = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                     3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5};
  ASSERT_EQ(ctx_4.Depth(), 2);
  ASSERT_EQ(ctx_4.TotalSegmentCount(), 4);
  ASSERT_EQ(ctx_4.StageSegmentCount(), 4);
  for (int64_t i = 0; i < 24; i++) {
    ASSERT_EQ(ctx_4.StageRank4ParallelId(i), ctx_4_ranks[i]);
    ASSERT_EQ(ctx_4.RankSet4ParallelId(i), ctx_4_sets[i]);
  }

  ReduceRankCtx ctx_6 = ReduceRankCtx().CtxWithScatter(6);
  std::vector<int64_t> ctx_6_ranks = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                                      0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
  std::vector<int64_t> ctx_6_sets = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                                     2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3};
  ASSERT_EQ(ctx_6.Depth(), 2);
  ASSERT_EQ(ctx_6.TotalSegmentCount(), 6);
  ASSERT_EQ(ctx_6.StageSegmentCount(), 6);
  for (int64_t i = 0; i < 24; i++) {
    ASSERT_EQ(ctx_6.StageRank4ParallelId(i), ctx_6_ranks[i]);
    ASSERT_EQ(ctx_6.RankSet4ParallelId(i), ctx_6_sets[i]);
  }

  ReduceRankCtx ctx_3_2 = ReduceRankCtx().CtxWithScatter(3).CtxWithScatter(2);
  std::vector<int64_t> ctx_3_2_ranks = {0, 0, 0, 1, 1, 1};
  std::vector<int64_t> ctx_3_2_sets = {0, 1, 2, 0, 1, 2};
  ASSERT_EQ(ctx_3_2.Depth(), 3);
  ASSERT_EQ(ctx_3_2.TotalSegmentCount(), 6);
  ASSERT_EQ(ctx_3_2.StageSegmentCount(), 2);
  for (int64_t i = 0; i < 6; i++) {
    ASSERT_EQ(ctx_3_2.StageRank4ParallelId(i), ctx_3_2_ranks[i]);
    ASSERT_EQ(ctx_3_2.RankSet4ParallelId(i), ctx_3_2_sets[i]);
  }

  ReduceRankCtx ctx_4_3_2 = ReduceRankCtx().CtxWithScatter(4).CtxWithScatter(3).CtxWithScatter(2);
  std::vector<int64_t> ctx_4_3_2_ranks = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> ctx_4_3_2_sets = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  ASSERT_EQ(ctx_4_3_2.Depth(), 4);
  ASSERT_EQ(ctx_4_3_2.TotalSegmentCount(), 24);
  ASSERT_EQ(ctx_4_3_2.StageSegmentCount(), 2);
  for (int64_t i = 0; i < 24; i++) {
    ASSERT_EQ(ctx_4_3_2.StageRank4ParallelId(i), ctx_4_3_2_ranks[i]);
    ASSERT_EQ(ctx_4_3_2.RankSet4ParallelId(i), ctx_4_3_2_sets[i]);
  }
}

}  // namespace oneflow