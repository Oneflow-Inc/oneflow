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
#include <string>
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

bool SubTskGphBuilderUtil::IsDeviceTypeCPUOrGPU(const ParallelDesc& parallel_desc) {
  return parallel_desc.device_type() == DeviceType::kCPU
         || parallel_desc.device_type() == DeviceType::kGPU;
}

std::vector<TensorSliceView> SubTskGphBuilderUtil::GetTensorSliceView(
    const int64_t parallel_num, const SbpParallel& sbp_parallel, const BlobDesc& blob_desc) {
  std::vector<Range> ranges(blob_desc.shape().NumAxes());
  FOR_RANGE(int64_t, i, 0, blob_desc.shape().NumAxes()) {
    ranges[i].mut_begin() = 0;
    ranges[i].mut_end() = blob_desc.shape().At(i);
  }
  std::vector<TensorSliceView> views;
  if (sbp_parallel.has_partial_sum_parallel() || sbp_parallel.has_broadcast_parallel()) {
    FOR_RANGE(int64_t, i, 0, parallel_num) { views.emplace_back(ranges); }
  } else if (sbp_parallel.has_split_parallel()) {
    const int64_t axis = sbp_parallel.split_parallel().axis();
    const BalancedSplitter bs(blob_desc.shape().At(axis), parallel_num);
    FOR_RANGE(int64_t, i, 0, parallel_num) {
      if (bs.At(i).size() == 0) {
        views.emplace_back();
      } else {
        ranges[axis] = bs.At(i);
        views.emplace_back(ranges);
      }
    }
  } else {
    UNIMPLEMENTED();
  }
  return views;
}

TensorSliceView SubTskGphBuilderUtil::GetBroadcastTensorSliceView(const BlobDesc& blob_desc) {
  return TensorSliceView(blob_desc.shape());
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

bool SubTskGphBuilderUtil::IsErrorBoxingNotSupported(const ErrorProto& error) {
  return error.has_boxing_error() && error.boxing_error() == BoxingError::kNotSupported;
}

int64_t SubTskGphBuilderUtil::GetDistance(const TaskNode* src, const TaskNode* dst) {
  if (src->machine_id() != dst->machine_id()) {
    return kDistanceDiffMachine;
  } else if (src->device_type() != dst->device_type()) {
    return kDistanceSameMachine;
  } else if (src->device_type() == DeviceType::kCPU) {
    return kDistanceSameDevice;
  } else {
    if (Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(src->thrd_id())
        == Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(dst->thrd_id())) {
      return kDistanceSameDevice;
    } else {
      return kDistanceSameMachine;
    }
  }
}

std::string SubTskGphBuilderUtil::SerializeSbpParallel(const SbpParallel& sbp_parallel) {
  std::string Serialized_sbp_parallel = "";
  if (sbp_parallel.has_broadcast_parallel()) {
    Serialized_sbp_parallel = "B";
  } else if (sbp_parallel.has_partial_sum_parallel()) {
    Serialized_sbp_parallel = "P";

  } else if (sbp_parallel.has_split_parallel()) {
    Serialized_sbp_parallel = "S(" + std::to_string(sbp_parallel.split_parallel().axis()) + ")";
  } else {
    UNIMPLEMENTED();
  }
  return Serialized_sbp_parallel;
}

std::string SubTskGphBuilderUtil::SerializeParallelDesc(const ParallelDesc& parallel_desc) {
  std::string Serialized_parallel_desc;
  Serialized_parallel_desc = PbMessage2TxtString(parallel_desc.parallel_conf());
  StringReplace(&Serialized_parallel_desc, '\n', ' ');
  Serialized_parallel_desc.pop_back();
  return Serialized_parallel_desc;
}

std::string SubTskGphBuilderUtil::SerializeLogicalBlobId(const LogicalBlobId& lbi) {
  std::string lbi_info = "";
  CHECK(lbi.has_op_name());
  lbi_info += "op_name: " + lbi.op_name() + " ";
  CHECK(lbi.has_blob_name());
  lbi_info += "blob_name: " + lbi.blob_name();
  if (lbi.has_is_packed_id()) {
    std::string is_packed_id = lbi.is_packed_id() ? "true" : "false";
    lbi_info += " is_packed_id: " + is_packed_id;
  }
  return lbi_info;
}

std::string SubTskGphBuilderUtil::GetBlobInfo4LogicalBlobDesc(const BlobDesc& logical_blob_desc) {
  std::string blob_desc_info = "dtype: TODO";

  auto dtype = logical_blob_desc.data_type();
  // blob_desc_info = blob_desc_info + dtype;

  auto shape_info = logical_blob_desc.shape().ToString();
  StringReplace(&shape_info, ',', ' ');
  blob_desc_info += " shape: " + shape_info;
  return blob_desc_info;
}

std::string SubTskGphBuilderUtil::SubTskGphBuilderStatus2String(
    const SubTskGphBuilderStatus& status) {
  std::string serialized_status("");
  serialized_status += status.src_op_name_ + ",";
  serialized_status += status.src_parallel_conf_ + ",";
  serialized_status += status.src_spb_parallel_ + ",";
  serialized_status += status.lbi_info_ + ",";
  serialized_status += status.logical_blob_desc_info_ + ",";
  serialized_status += status.boxing_type_ + ",";
  serialized_status += status.dst_op_name_ + ",";
  serialized_status += status.dst_parallel_conf_ + ",";
  serialized_status += status.dst_sbp_parallel_ + "\n";
  return serialized_status;
}

Maybe<SubTskGphBuilderStatus> SubTskGphBuilderUtil::BuildBoxingLogInfo(
    const CompTaskNode* src_node, const CompTaskNode* dst_node,
    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
    const SbpParallel& src_sbp_parallel, const SbpParallel& dst_sbp_parallel,
    const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc, const std::string& boxing_type) {
  SubTskGphBuilderStatus status;

  status.src_op_name_ = src_node->logical_node()->op_vec().at(0)->op_name();

  std::string parallel_desc_info = SubTskGphBuilderUtil::SerializeParallelDesc(src_parallel_desc);
  status.src_parallel_conf_ = parallel_desc_info;

  std::string sbp_parallel_info = SubTskGphBuilderUtil::SerializeSbpParallel(src_sbp_parallel);
  status.src_spb_parallel_ = sbp_parallel_info;

  status.lbi_info_ = SubTskGphBuilderUtil::SerializeLogicalBlobId(lbi);

  status.logical_blob_desc_info_ =
      SubTskGphBuilderUtil::GetBlobInfo4LogicalBlobDesc(logical_blob_desc);

  status.boxing_type_ = boxing_type;

  status.dst_op_name_ = dst_node->logical_node()->op_vec().at(0)->op_name() + ",";

  parallel_desc_info = SubTskGphBuilderUtil::SerializeParallelDesc(dst_parallel_desc);
  status.dst_parallel_conf_ = parallel_desc_info;

  sbp_parallel_info = SubTskGphBuilderUtil::SerializeSbpParallel(dst_sbp_parallel);
  status.dst_sbp_parallel_ = sbp_parallel_info;

  return status;
}

}  // namespace oneflow
