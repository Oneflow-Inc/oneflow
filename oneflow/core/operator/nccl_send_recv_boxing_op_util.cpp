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
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/operator/nccl_send_recv_boxing_op_util.h"

namespace oneflow {

namespace {

// Go through all the ranks while transfer between two nd sbps with no PartialSum under the same
// placement.
// NOTE: We need to make sure no partial sums in the sbps of the producer and consumer.
void DfsTraverseRanks4NdSbp(
    int32_t depth, std::vector<int64_t>& in_parallel_ids,
    const std::vector<int64_t>& out_parallel_ids, const Shape& parallel_hierarchy,
    const NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE>& hierarchy_index_helper,
    const NdSbp& in_nd_sbp, const std::function<void(int32_t, int32_t)>& visit) {
  if (depth >= parallel_hierarchy.NumAxes()) {
    visit(hierarchy_index_helper.NdIndexToOffset(out_parallel_ids.data(),
                                                 parallel_hierarchy.NumAxes()),
          hierarchy_index_helper.NdIndexToOffset(in_parallel_ids.data(),
                                                 parallel_hierarchy.NumAxes()));
    return;
  }
  if (in_nd_sbp.sbp_parallel(depth).has_broadcast_parallel()) {
    // If Broadcast in the sbp of the producer, only visit those ranks with the same id as the
    // current rank along the depth-dimension.
    in_parallel_ids[depth] = out_parallel_ids[depth];
    DfsTraverseRanks4NdSbp(depth + 1, in_parallel_ids, out_parallel_ids, parallel_hierarchy,
                           hierarchy_index_helper, in_nd_sbp, visit);
  } else {
    // If Split or PartialSum, go through all the ranks along the depth-dimension.
    for (int64_t i = 0; i < parallel_hierarchy.dim_vec().at(depth); i++) {
      in_parallel_ids[depth] = i;
      DfsTraverseRanks4NdSbp(depth + 1, in_parallel_ids, out_parallel_ids, parallel_hierarchy,
                             hierarchy_index_helper, in_nd_sbp, visit);
    }
  }
}

void DfsTraverse4NdSbp(int64_t out_id, const std::shared_ptr<Shape> parallel_hierarchy,
                       const NdSbp& in_nd_sbp, const std::function<void(int32_t, int32_t)>& visit) {
  int32_t hierarchy_dimension = parallel_hierarchy->NumAxes();
  const NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE> hierarchy_index_helper(
      parallel_hierarchy->dim_vec().data(), hierarchy_dimension);
  std::vector<int64_t> in_parallel_ids(hierarchy_dimension);
  std::vector<int64_t> out_parallel_ids(hierarchy_dimension);
  hierarchy_index_helper.OffsetToNdIndex(out_id, out_parallel_ids.data(), hierarchy_dimension);
  DfsTraverseRanks4NdSbp(0, in_parallel_ids, out_parallel_ids, *parallel_hierarchy,
                         hierarchy_index_helper, in_nd_sbp, visit);
}

bool NdSbpNoPartialParallel(const NdSbp& nd_sbp) {
  CHECK_GT(nd_sbp.sbp_parallel_size(), 0);
  FOR_RANGE(int64_t, i, 0, nd_sbp.sbp_parallel_size()) {
    if (nd_sbp.sbp_parallel(i).has_partial_sum_parallel()) { return false; }
  }
  return true;
}

}  // namespace

void GetSendRecvIntersection(int64_t parallel_id, const std::shared_ptr<Shape> parallel_hierarchy,
                             const NdSbp& src_nd_sbp, const NdSbp& dst_nd_sbp,
                             const Shape& logical_shape,
                             std::vector<TensorSliceView>* src_send_intersections,
                             std::vector<TensorSliceView>* dst_recv_intersections) {
  CHECK(parallel_hierarchy);
  const int64_t parallel_num = parallel_hierarchy->elem_cnt();
  CHECK_LT(parallel_id, parallel_num);

  const std::vector<TensorSliceView>& out_slices =
      GetTensorSliceView(*parallel_hierarchy, dst_nd_sbp, logical_shape);
  const std::vector<TensorSliceView>& in_slices =
      GetTensorSliceView(*parallel_hierarchy, src_nd_sbp, logical_shape);

  // cur_out_slice recv from
  dst_recv_intersections->resize(parallel_num);
  const TensorSliceView& cur_rank_out_slice = out_slices.at(parallel_id);
  const auto& add_to_dst_recv_intersections = [&](int32_t out_id, int32_t in_id) {
    CHECK_EQ(out_id, parallel_id);
    const TensorSliceView& in_slice = in_slices.at(in_id);
    const TensorSliceView& intersection = cur_rank_out_slice.Intersect(in_slice);
    dst_recv_intersections->at(in_id) = intersection;
  };
  DfsTraverse4NdSbp(parallel_id, parallel_hierarchy, src_nd_sbp, add_to_dst_recv_intersections);

  // cur_in_slice send to
  src_send_intersections->resize(parallel_num);
  const TensorSliceView& cur_rank_in_slice = in_slices.at(parallel_id);
  const auto& add_to_src_send_intersections = [&](int32_t out_id, int32_t in_id) {
    if (in_id != parallel_id) { return; }
    const TensorSliceView& out_slice = out_slices.at(out_id);
    const TensorSliceView& intersection = out_slice.Intersect(cur_rank_in_slice);
    src_send_intersections->at(out_id) = intersection;
  };
  for (int64_t i = 0; i < parallel_num; ++i) {
    DfsTraverse4NdSbp(i, parallel_hierarchy, src_nd_sbp, add_to_src_send_intersections);
  }
}

}  // namespace oneflow
