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
    const std::vector<int64_t>& out_parallel_ids, const Shape& in_parallel_hierarchy,
    const NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE>& in_hierarchy_index_helper,
    const NdSbp& in_nd_sbp, const std::function<void(int32_t)>& visit) {
  if (depth >= in_parallel_hierarchy.NumAxes()) {
    visit(in_hierarchy_index_helper.NdIndexToOffset(in_parallel_ids.data(),
                                                    in_parallel_hierarchy.NumAxes()));
    return;
  }
  if (in_nd_sbp.sbp_parallel(depth).has_broadcast_parallel()) {
    // If Broadcast in the sbp of the producer, only visit those ranks with the same id as the
    // current rank along the depth-dimension.
    in_parallel_ids[depth] = out_parallel_ids[depth];
    DfsTraverseRanks4NdSbp(depth + 1, in_parallel_ids, out_parallel_ids, in_parallel_hierarchy,
                           in_hierarchy_index_helper, in_nd_sbp, visit);
  } else {
    // If Split or PartialSum, go through all the ranks along the depth-dimension.
    for (int64_t i = 0; i < in_parallel_hierarchy.dim_vec().at(depth); i++) {
      in_parallel_ids[depth] = i;
      DfsTraverseRanks4NdSbp(depth + 1, in_parallel_ids, out_parallel_ids, in_parallel_hierarchy,
                             in_hierarchy_index_helper, in_nd_sbp, visit);
    }
  }
}

bool NdSbpNoPartialParallel(const NdSbp& nd_sbp) {
  CHECK_GT(nd_sbp.sbp_parallel_size(), 0);
  FOR_RANGE(int64_t, i, 0, nd_sbp.sbp_parallel_size()) {
    if (nd_sbp.sbp_parallel(i).has_partial_sum_parallel()) { return false; }
  }
  return true;
}

}  // namespace

int64_t GetMappedParallelId(const int64_t from_parallel_id, const ParallelDesc& from_parallel_desc,
                            const ParallelDesc& to_parallel_desc) {
  const int64_t machine_id = CHECK_JUST(from_parallel_desc.MachineId4ParallelId(from_parallel_id));
  const int64_t device_index = CHECK_JUST(from_parallel_desc.DeviceId4ParallelId(from_parallel_id));
  if (to_parallel_desc.Containing(machine_id, device_index)) {
    return CHECK_JUST(to_parallel_desc.ParallelId4MachineDeviceId(machine_id, device_index));
  } else {
    return -1;
  }
}

void GetRankSendRecvIntersection(int64_t parallel_id, const ParallelDesc& parallel_desc,
                                 const ParallelDesc& in_parallel_desc,
                                 const ParallelDesc& out_parallel_desc, const NdSbp& in_nd_sbp,
                                 const NdSbp& out_nd_sbp, const Shape& logical_shape,
                                 std::vector<TensorSliceView>* send_intersections,
                                 std::vector<TensorSliceView>* recv_intersections) {
  const int64_t parallel_num = parallel_desc.parallel_num();
  CHECK_LT(parallel_id, parallel_num);

  const std::vector<TensorSliceView>& in_slices =
      GetTensorSliceView(*in_parallel_desc.hierarchy(), in_nd_sbp, logical_shape);
  const std::vector<TensorSliceView>& out_slices =
      GetTensorSliceView(*out_parallel_desc.hierarchy(), out_nd_sbp, logical_shape);

  const auto& in_parallel_hierarchy = in_parallel_desc.hierarchy();
  int32_t in_hierarchy_dimension = in_parallel_hierarchy->NumAxes();
  const NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE> in_hierarchy_index_helper(
      in_parallel_hierarchy->dim_vec().data(), in_hierarchy_dimension);

  const int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(parallel_id));
  const int64_t device_index = CHECK_JUST(parallel_desc.DeviceId4ParallelId(parallel_id));
  const int64_t in_parallel_num = in_parallel_desc.parallel_num();
  const int64_t out_parallel_num = out_parallel_desc.parallel_num();
  // cur rank recv from
  // cur rank has output
  if (out_parallel_desc.Containing(machine_id, device_index)) {
    recv_intersections->resize(parallel_num);
    int64_t out_id =
        CHECK_JUST(out_parallel_desc.ParallelId4MachineDeviceId(machine_id, device_index));
    const TensorSliceView& cur_rank_out_slice = out_slices.at(out_id);
    const auto& add_to_recv_intersections = [&](int32_t send_id) {
      const TensorSliceView& in_slice = in_slices.at(send_id);
      const TensorSliceView& intersection = cur_rank_out_slice.Intersect(in_slice);
      if (intersection.IsEmpty()) { return; }
      const int64_t merged_id = GetMappedParallelId(send_id, in_parallel_desc, parallel_desc);
      recv_intersections->at(merged_id) = intersection;
    };
    int64_t corresponding_in_id = 0;
    // For example [[0, 1], [2, 3]] -> [[1, 3], [5, 6]]
    if (in_parallel_desc.Containing(machine_id, device_index)) {
      // 1 and 3 are in [[0, 1], [2, 3]], use the same id in the producer parallel description
      // The id of 1 is (0, 1), the id of 3 is (1, 1)
      corresponding_in_id =
          CHECK_JUST(in_parallel_desc.ParallelId4MachineDeviceId(machine_id, device_index));
    } else {
      // 5 and 7 are not in [[0, 1], [2, 3]]
      // Then the id does not matter
      corresponding_in_id = out_id % in_parallel_num;
    }
    std::vector<int64_t> in_parallel_ids(in_hierarchy_dimension);
    // The corresponding parallel id of a consumer rank in the producer parallel description
    std::vector<int64_t> out_parallel_ids(in_hierarchy_dimension);
    in_hierarchy_index_helper.OffsetToNdIndex(corresponding_in_id, out_parallel_ids.data(),
                                              in_hierarchy_dimension);
    DfsTraverseRanks4NdSbp(0, in_parallel_ids, out_parallel_ids, *in_parallel_hierarchy,
                           in_hierarchy_index_helper, in_nd_sbp, add_to_recv_intersections);
  }

  // cur rank send to
  if (in_parallel_desc.Containing(machine_id, device_index)) {
    send_intersections->resize(parallel_num);
    int64_t in_id =
        CHECK_JUST(in_parallel_desc.ParallelId4MachineDeviceId(machine_id, device_index));
    const TensorSliceView& cur_rank_in_slice = in_slices.at(in_id);
    for (int64_t recv_i = 0; recv_i < out_parallel_num; ++recv_i) {
      const auto& add_to_send_intersections = [&](int32_t send_id) {
        if (send_id != in_id) { return; }
        const TensorSliceView& out_slice = out_slices.at(recv_i);
        const TensorSliceView& intersection = out_slice.Intersect(cur_rank_in_slice);
        if (intersection.IsEmpty()) { return; }
        const int64_t merged_id = GetMappedParallelId(recv_i, out_parallel_desc, parallel_desc);
        send_intersections->at(merged_id) = intersection;
      };
      int64_t out_device_id = CHECK_JUST(out_parallel_desc.DeviceId4ParallelId(recv_i));
      int64_t out_machine_id = CHECK_JUST(out_parallel_desc.MachineId4ParallelId(recv_i));
      int64_t corresponding_in_id = 0;
      // For example [[0, 1], [2, 3]] -> [[1, 3], [5, 6]]
      if (in_parallel_desc.Containing(out_machine_id, out_device_id)) {
        // 1 and 3 are in [[0, 1], [2, 3]], use the same id in the producer parallel description
        // The id of 1 is (0, 1), the id of 3 is (1, 1)
        corresponding_in_id =
            CHECK_JUST(in_parallel_desc.ParallelId4MachineDeviceId(out_machine_id, out_device_id));
      } else {
        // 5 and 7 are not in [[0, 1], [2, 3]]
        // Then the id does not matter
        corresponding_in_id = recv_i % in_parallel_num;
      }
      std::vector<int64_t> in_parallel_ids(in_hierarchy_dimension);
      // The corresponding parallel id of a consumer rank in the producer parallel description
      std::vector<int64_t> out_parallel_ids(in_hierarchy_dimension);
      in_hierarchy_index_helper.OffsetToNdIndex(corresponding_in_id, out_parallel_ids.data(),
                                                in_hierarchy_dimension);
      DfsTraverseRanks4NdSbp(0, in_parallel_ids, out_parallel_ids, *in_parallel_hierarchy,
                             in_hierarchy_index_helper, in_nd_sbp, add_to_send_intersections);
    }
  }
}

}  // namespace oneflow
