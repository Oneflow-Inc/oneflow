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
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

bool SubTskGphBuilderUtil::IsDeviceTypeCPUOrGPU(const ParallelDesc& parallel_desc) {
  return parallel_desc.device_type() == DeviceType::kCPU
         || parallel_desc.device_type() == DeviceType::kGPU;
}

bool SubTskGphBuilderUtil::HasEmptySliceIfSplit(int64_t parallel_num,
                                                const SbpParallel& sbp_parallel,
                                                const BlobDesc& blob_desc) {
  if (sbp_parallel.has_split_parallel()) {
    return blob_desc.shape().At(sbp_parallel.split_parallel().axis()) < parallel_num;
  } else {
    return false;
  }
}

bool SubTskGphBuilderUtil::IsOnSameGPU(const TaskNode* lhs, const TaskNode* rhs) {
  return lhs->machine_id() == rhs->machine_id() && lhs->device_type() == DeviceType::kGPU
         && rhs->device_type() == DeviceType::kGPU && lhs->GpuPhyId() == rhs->GpuPhyId();
}

bool SubTskGphBuilderUtil::IsBoxingS2S(const SbpParallel& src, const SbpParallel& dst) {
  return src.has_split_parallel() && dst.has_split_parallel();
}

bool SubTskGphBuilderUtil::IsBoxingS2B(const SbpParallel& src, const SbpParallel& dst) {
  return src.has_split_parallel() && dst.has_broadcast_parallel();
}

bool SubTskGphBuilderUtil::IsBoxingP2S(const SbpParallel& src, const SbpParallel& dst) {
  return src.has_partial_sum_parallel() && dst.has_split_parallel();
}

bool SubTskGphBuilderUtil::IsBoxingP2B(const SbpParallel& src, const SbpParallel& dst) {
  return src.has_partial_sum_parallel() && dst.has_broadcast_parallel();
}

bool SubTskGphBuilderUtil::IsBoxingB2B(const SbpParallel& src, const SbpParallel& dst) {
  return src.has_broadcast_parallel() && dst.has_broadcast_parallel();
}

bool SubTskGphBuilderUtil::IsBoxingB2S(const SbpParallel& src, const SbpParallel& dst) {
  return src.has_broadcast_parallel() && dst.has_split_parallel();
}

bool SubTskGphBuilderUtil::BlobHasDynamicShape(const BlobDesc& blob_desc) {
  return blob_desc.is_dynamic();
}

bool SubTskGphBuilderUtil::IsErrorBoxingNotSupported(const cfg::ErrorProto& error) {
  return error.has_boxing_not_supported_error();
}

int64_t SubTskGphBuilderUtil::GetDistance(
    const int64_t src_machine_id, const int64_t src_dev_phy_id, const DeviceType src_device_type,
    const int64_t dst_machine_id, const int64_t dst_dev_phy_id, const DeviceType dst_device_type) {
  if (src_machine_id != dst_machine_id) {
    return kDistanceDiffMachine;
  } else if (src_device_type != dst_device_type) {
    return kDistanceSameMachine;
  } else if (src_device_type == DeviceType::kCPU) {
    return kDistanceSameDevice;
  } else {
    if (src_dev_phy_id == dst_dev_phy_id) {
      return kDistanceSameDevice;
    } else {
      return kDistanceSameMachine;
    }
  }
}

int64_t SubTskGphBuilderUtil::GetDistance(const ParallelDesc& src_parallel_desc,
                                          const int64_t src_parallel_id,
                                          const ParallelDesc& dst_parallel_desc,
                                          const int64_t dst_parallel_id) {
  const int64_t src_machine_id =
      CHECK_JUST(src_parallel_desc.MachineId4ParallelId(src_parallel_id));
  const int64_t src_dev_phy_id = CHECK_JUST(src_parallel_desc.DeviceId4ParallelId(src_parallel_id));
  const int64_t dst_machine_id =
      CHECK_JUST(dst_parallel_desc.MachineId4ParallelId(dst_parallel_id));
  const int64_t dst_dev_phy_id = CHECK_JUST(dst_parallel_desc.DeviceId4ParallelId(dst_parallel_id));
  return GetDistance(src_machine_id, src_dev_phy_id, src_parallel_desc.device_type(),
                     dst_machine_id, dst_dev_phy_id, dst_parallel_desc.device_type());
}

int64_t SubTskGphBuilderUtil::GetDistance(const TaskNode* src, const TaskNode* dst) {
  const auto GetDevPhyId = [](const DeviceType device_type, const int64_t thrd_id) -> int64_t {
    if (device_type == DeviceType::kGPU) {
      return Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(thrd_id);
    } else if (device_type == DeviceType::kCPU) {
      return 0;
    } else {
      UNIMPLEMENTED();
    }
  };
  const DeviceType src_device_type = src->device_type();
  const int64_t src_dev_phy_id = GetDevPhyId(src_device_type, src->thrd_id());
  const DeviceType dst_device_type = dst->device_type();
  const int64_t dst_dev_phy_id = GetDevPhyId(dst_device_type, dst->thrd_id());
  return GetDistance(src->machine_id(), src_dev_phy_id, src_device_type, dst->machine_id(),
                     dst_dev_phy_id, dst_device_type);
}

int64_t SubTskGphBuilderUtil::FindNearestSrcParallelId(const ParallelDesc& from_parallel_desc,
                                                       const ParallelDesc& to_parallel_desc,
                                                       const int64_t to_parallel_id) {
  int64_t nearest_from_parallel_idx = -1;
  int64_t nearest_distance = SubTskGphBuilderUtil::kDistanceMax;
  for (int64_t i = 0; i < from_parallel_desc.parallel_num(); ++i) {
    const int64_t distance =
        SubTskGphBuilderUtil::GetDistance(from_parallel_desc, i, to_parallel_desc, to_parallel_id);
    if (distance < nearest_distance) {
      nearest_from_parallel_idx = i;
      nearest_distance = distance;
    }
  }
  CHECK_NE(nearest_from_parallel_idx, -1);
  return nearest_from_parallel_idx;
}

}  // namespace oneflow
